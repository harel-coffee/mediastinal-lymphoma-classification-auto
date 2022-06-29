#  Run with:
#    python model_multi_classifier.py -m log-reg -s 60/40 -t rec90


import os
import pickle
import numpy as np

from tqdm     import tqdm
from argparse import ArgumentParser

import optuna
optuna.logging.set_verbosity ( optuna.logging.ERROR )   # silence Optuna during trials study

import warnings
warnings.filterwarnings ( "ignore", category = RuntimeWarning )

from sklearn.model_selection   import StratifiedShuffleSplit
from sklearn.preprocessing     import MinMaxScaler
from imblearn.over_sampling    import SMOTE
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import SVC
from sklearn.gaussian_process  import GaussianProcessClassifier
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics           import roc_auc_score, confusion_matrix, roc_curve

from utils import custom_predictions, multiclass_predictions, plot_conf_matrices, plot_multi_prf_histos

LABELS = ["cHL", "GZL", "PMBCL"]

#   +-------------------+
#   |   Options setup   |
#   +-------------------+

MODELS = [ "log-reg", "lin-svm", "gaus-proc", "rnd-frs", "grad-bdt", "suv-max" ]

parser = ArgumentParser ( description = "training script" )
parser . add_argument ( "-m" , "--model"     , required = True , choices = MODELS )
parser . add_argument ( "-s" , "--split"     , default  = "60/40" )
parser . add_argument ( "-t" , "--threshold" , default  = "rec90" )
parser . add_argument ( "-J" , "--NUM_JOBS"  , default  = 1 )
args = parser . parse_args()

if len ( args.split.split("/") ) == 2:
  test_size = 0.5 * float(args.split.split("/")[-1]) / 100
  val_size  = test_size
  val_size /= ( 1 - test_size )   # w.r.t. new dataset size
elif len ( args.split.split("/") ) == 3:
  test_size = float(args.split.split("/")[2]) / 100
  val_size  = float(args.split.split("/")[1]) / 100
  val_size /= ( 1 - test_size )   # w.r.t. new dataset size
else:
  raise ValueError (f"The splitting ratios should be passed as 'XX/YY/ZZ', where XX% is "
                    f"the percentage of data used for training, while YY% and ZZ% are "
                    f"the ones used for validation and testing respectively.")

if "rec" in args.threshold:
  rec_score  = float(args.threshold.split("rec")[-1]) / 100
  prec_score = None
elif "prec" in args.threshold:
  rec_score  = None
  prec_score = float(args.threshold.split("prec")[-1]) / 100
else:
  raise ValueError (f"The rule for custom predictions should be passed as 'recXX' where "
                    f"XX% is the minimum recall score required, or as 'precYY' where YY% "
                    f"is the minimum precision score required.")

NUM_JOBS = int ( args.NUM_JOBS )
if NUM_JOBS < 1:
  raise ValueError ("The number of jobs should be greater or equal to 1.")

#   +------------------+
#   |   Optuna setup   |
#   +------------------+

def optuna_study ( model_name  : str ,
                   storage_dir : str ,
                   objective   : tuple ,
                   n_trials    : int = 10 ,
                   directions  : list = [ "minimize" , "minimize" ] , 
                   load_if_exists : bool = False  ) -> optuna.study.Study:
  storage_path = "{}/{}.db" . format (storage_dir, model_name)
  storage_name = "sqlite:///{}" . format (storage_path)  

  if load_if_exists:
    pass
  elif not ( load_if_exists ) and os.path.isfile ( storage_path ):
    os.remove ( storage_path )

  study = optuna.create_study ( study_name = model_name   ,
                                storage    = storage_name ,
                                load_if_exists = load_if_exists ,
                                directions = directions )

  study . optimize ( objective, n_trials = n_trials )

  return study

#   +------------------+
#   |   Data loading   |
#   +------------------+

data_dir  = "./data"
data_file = "db_mediastinalbulky_reduced.pkl" 
file_path = os.path.join ( data_dir, data_file )

with open (file_path, "rb") as file:
  data = pickle.load (file)

#   +------------------------------+
#   |   Input/output preparation   |
#   +------------------------------+

cols = list ( data.columns )
X_cols = cols[2:]
y_cols = "lymphoma_type"

X = data.query("lymphoma_type != 2")[X_cols] . to_numpy()
y = data.query("lymphoma_type != 2")[y_cols] . to_numpy() . flatten()
y = ( y == 3 )   # PMBCL/cHL classification

X_gz = data.query("lymphoma_type == 2")[X_cols] . to_numpy()
y_gz = data.query("lymphoma_type == 2")[y_cols] . to_numpy() . flatten()

#   +------------------------+
#   |   Sub-sample studies   |
#   +------------------------+

conf_matrices = [ list() , list() ]   # container for 3x3 confusion matrices
tprs = [ list() , list() ]            # container for one-vs-all TPRs
tnrs = [ list() , list() ]            # container for one-vs-all TNRs
roc_curves = [ list() , list() ]      # container for one-vs-all ROC curve variables
auc_scores = [ list() , list() ]      # container for one-vs-all AUC score values

## initial control values
optimized = False
append_to_roc = [ True , True ]
n_roc_points  = [ -1   , -1   ]

for i in tqdm(range(300)):

  #   +--------------------------+
  #   |   Train/test splitting   |
  #   +--------------------------+

  sss = StratifiedShuffleSplit ( n_splits = 1, test_size = test_size )
  for idx_train, idx_test in sss . split ( X, y ):
    X_train , y_train = X[idx_train] , y[idx_train]
    X_test  , y_test  = X[idx_test]  , y[idx_test] 

  #   +------------------------+
  #   |   Data preprocessing   |
  #   +------------------------+

  scaler_train = MinMaxScaler()
  X_train = scaler_train.fit_transform (X_train)

  scaler_test = MinMaxScaler()
  X_test = scaler_test.fit_transform (X_test)

  scaler_gz = MinMaxScaler()
  X_gz = scaler_gz.fit_transform (X_gz)

  #   +------------------------------+
  #   |   Hyperparams optimization   |
  #   +------------------------------+

  ## LOGISTIC REGRESSION
  if args.model == "log-reg":
    best_model = LogisticRegression ( n_jobs = NUM_JOBS )

  ## LINEAR SVM
  elif args.model == "lin-svm":
    best_model = SVC ( kernel = "linear", probability = True )

  ## GAUSSIAN PROCESS
  elif args.model == "gaus-proc":
    best_model = GaussianProcessClassifier ( n_jobs = NUM_JOBS )

  ## RANDOM FOREST
  elif args.model == "rnd-frs":
    def objective (trial):
      ## train/val splitting
      sss = StratifiedShuffleSplit ( n_splits = 1, test_size = val_size )
      for idx_train, idx_val in sss . split ( X_train, y_train ):
        X_trn , y_trn = X_train[idx_train] , y_train[idx_train]
        X_val , y_val = X_train[idx_val]   , y_train[idx_val] 

      sm = SMOTE()   # oversampling technique
      X_trn_res, y_trn_res = sm.fit_resample ( X_trn , y_trn )

      ## hyperparams to optimize
      max_depth        = trial . suggest_int   ( "max_depth"        ,    1 ,  10 )
      min_samples_leaf = trial . suggest_float ( "min_samples_leaf" , 1e-8 , 0.5 )

      ## model to optimize
      model = RandomForestClassifier ( n_estimators     = 100 ,
                                       max_depth        = max_depth ,
                                       min_samples_leaf = min_samples_leaf ,
                                       max_features     = None ,
                                       n_jobs           = NUM_JOBS )

      model.fit (X_trn_res, y_trn_res)

      ## scores to optimize
      y_scores_trn = model.predict_proba (X_trn) [:,1]
      y_scores_val = model.predict_proba (X_val) [:,1]

      auc_trn = roc_auc_score ( y_trn, y_scores_trn )
      auc_val = roc_auc_score ( y_val, y_scores_val )

      return auc_val , np.abs ( auc_trn - auc_val ) / auc_val   # AUC , over-fitting

    if not optimized:
      study = optuna_study ( model_name  = "rnd_forest_clf" ,
                             storage_dir = "./storage" ,
                             objective   = objective ,
                             n_trials    = 50 ,
                             directions  = [ "maximize" , "minimize" ] ,
                             load_if_exists = False )

      df = study . trials_dataframe ( attrs = ("params", "values") ) 
      df = df [ (df["values_0"] > 0.5) & (df["values_0"] < 1.0)  ]
      df_head = df . sort_values ( by = "values_1", ascending = True ) [:10]
      print ( df_head )

      optimized = True

    best_model = RandomForestClassifier ( n_estimators     = 100 ,
                                          max_depth        = df_head["params_max_depth"].values[0] ,
                                          min_samples_leaf = df_head["params_min_samples_leaf"].values[0] ,
                                          max_features     = None ,
                                          n_jobs           = NUM_JOBS )

  ## GRADIENT BDT
  elif args.model == "grad-bdt":
    def objective (trial):
      ## train/val splitting
      sss = StratifiedShuffleSplit ( n_splits = 1, test_size = val_size )
      for idx_train, idx_val in sss . split ( X_train, y_train ):
        X_trn , y_trn = X_train[idx_train] , y_train[idx_train]
        X_val , y_val = X_train[idx_val]   , y_train[idx_val] 

      sm = SMOTE()   # oversampling technique
      X_trn_res, y_trn_res = sm.fit_resample ( X_trn , y_trn )

      ## hyperparams to optimize
      learn_rate       = trial . suggest_float ( "learn_rate"       , 5e-2 , 5e-1 , log = True  )
      max_depth        = trial . suggest_int   ( "max_depth"        ,    1 ,   10 , log = False )
      min_samples_leaf = trial . suggest_float ( "min_samples_leaf" , 1e-8 , 5e-1 , log = False )

      ## model to optimize
      model = GradientBoostingClassifier ( learning_rate    = learn_rate , 
                                           n_estimators     = 100        , 
                                           max_depth        = max_depth  ,
                                           min_samples_leaf = min_samples_leaf ,
                                           max_features     = None )

      model.fit (X_trn_res, y_trn_res)
      
      ## scores to optimize
      y_scores_trn = model.predict_proba (X_trn) [:,1]
      y_scores_val = model.predict_proba (X_val) [:,1]

      auc_trn = roc_auc_score ( y_trn, y_scores_trn )
      auc_val = roc_auc_score ( y_val, y_scores_val )

      return auc_val , np.abs ( auc_trn - auc_val ) / auc_val   # AUC , over-fitting

    if not optimized:
      study = optuna_study ( model_name  = "grad_bdt_clf" ,
                             storage_dir = "./storage" ,
                             objective   = objective ,
                             n_trials    = 50 ,
                             directions  = [ "maximize" , "minimize" ] ,
                             load_if_exists = False )
      
      df = study . trials_dataframe ( attrs = ("params", "values") )
      df = df [ (df["values_0"] > 0.5) & (df["values_0"] < 1.0)  ]
      df_head = df . sort_values ( by = "values_1", ascending = True ) [:10]
      print ( df_head )

      optimized = True

    best_model = GradientBoostingClassifier ( learning_rate    = df_head["params_learn_rate"].values[0] , 
                                              n_estimators     = 100 , 
                                              max_depth        = df_head["params_max_depth"].values[0]  ,
                                              min_samples_leaf = df_head["params_min_samples_leaf"].values[0] ,
                                              max_features     = None )

  ## SUVMAX-BASED CLASSIFIER
  elif args.model == "suv-max":
    best_model = None

  #   +---------------------------+
  #   |   Multiclass boundaries   |
  #   +---------------------------+

  def get_decision_boundaries ( y_scores : np.ndarray, threshold : float, width : float = 0.0 ) -> tuple:
    hist, bins = np.histogram ( y_scores[:,1], bins = 20 )
    cumsum  = np.cumsum ( hist.astype(np.float32) )
    cumsum /= cumsum[-1]

    scores = ( bins[1:] + bins[:-1] ) / 2.0
    th_cumsum = np.interp ( threshold, scores, cumsum )   # cumulative value of the threshold

    th_cumsum_min = th_cumsum - width / 2.0
    score_min = np.interp (th_cumsum_min, cumsum, scores)

    th_cumsum_max = th_cumsum + width / 2.0
    score_max = np.interp (th_cumsum_max, cumsum, scores)
    return score_min, score_max

  #   +-----------------------------------------+
  #   |   Model performance on train/test set   |
  #   +-----------------------------------------+

  if best_model is not None:
    sm = SMOTE()   # oversampling technique
    X_train_res, y_train_res = sm.fit_resample ( X_train , y_train )

    ## model training
    best_model . fit (X_train_res, y_train_res)

  ## combine the datasets
  X_train_comb = np.concatenate ( [ X_train, X_gz ] )
  y_train_comb = np.concatenate ( [ np.where(y_train, 3, 1), y_gz ] )

  X_test_comb = np.concatenate ( [ X_test, X_gz ] )
  y_test_comb = np.concatenate ( [ np.where(y_test, 3, 1), y_gz ] )

  ## model predictions
  y_scores_train = best_model.predict_proba ( X_train ) if (best_model is not None) else np.c_ [ 1 - X_train[:,0] , X_train[:,0] ]
  _, threshold = custom_predictions ( y_true = y_train , 
                                      y_scores = y_scores_train , 
                                      recall_score = rec_score  ,
                                      precision_score = prec_score )   

  y_scores_train_comb = best_model.predict_proba ( X_train_comb ) if (best_model is not None) else np.c_ [ 1 - X_train_comb[:,0] , X_train_comb[:,0] ]
  y_pred_train = multiclass_predictions ( y_true = y_train_comb ,
                                          y_scores = y_scores_train_comb ,
                                          boundaries = get_decision_boundaries ( y_scores_train , 
                                                                                 threshold , 
                                                                                 len(y_gz) / len(y_train_comb) ) )   # pred for the true train-set

  y_scores_test_comb = best_model.predict_proba ( X_test_comb ) if (best_model is not None) else np.c_ [ 1 - X_test_comb[:,0] , X_test_comb[:,0] ]
  y_pred_test = multiclass_predictions ( y_true = y_test_comb ,
                                         y_scores = y_scores_test_comb ,
                                         boundaries = get_decision_boundaries ( y_scores_train , 
                                                                                threshold , 
                                                                                len(y_gz) / len(y_train_comb) ) )   # pred for the test-set

  ## model performances
  multi_conf_matrix_train = confusion_matrix ( y_train_comb, y_pred_train )
  bin_conf_matrix_train_2 = confusion_matrix ( (y_train_comb == 3), (y_scores_train_comb[:,1] >= threshold) )
  tpr_train_2 = bin_conf_matrix_train_2[1,1] / np.sum ( bin_conf_matrix_train_2[1,:] )
  tnr_train_2 = bin_conf_matrix_train_2[0,0] / np.sum ( bin_conf_matrix_train_2[0,:] )
  bin_conf_matrix_train_1 = confusion_matrix ( (y_train_comb == 2), (y_scores_train_comb[:,1] >= threshold) )
  tpr_train_1 = bin_conf_matrix_train_1[1,1] / np.sum ( bin_conf_matrix_train_1[1,:] )
  tnr_train_1 = bin_conf_matrix_train_1[0,0] / np.sum ( bin_conf_matrix_train_1[0,:] )
  conf_matrices[0] . append ( multi_conf_matrix_train )   # add to the relative container
  tprs[0] . append ( [tpr_train_2, tpr_train_1] )         # add to the relative container
  tnrs[0] . append ( [tnr_train_2, tnr_train_1] )         # add to the relative container

  multi_conf_matrix_test = confusion_matrix ( y_test_comb, y_pred_test )
  bin_conf_matrix_test_2 = confusion_matrix ( (y_test_comb == 3), (y_scores_test_comb[:,1] >= threshold) )
  tpr_test_2 = bin_conf_matrix_test_2[1,1] / np.sum ( bin_conf_matrix_test_2[1,:] )
  tnr_test_2 = bin_conf_matrix_test_2[0,0] / np.sum ( bin_conf_matrix_test_2[0,:] )
  bin_conf_matrix_test_1 = confusion_matrix ( (y_test_comb == 2), (y_scores_test_comb[:,1] >= threshold) )
  tpr_test_1 = bin_conf_matrix_test_1[1,1] / np.sum ( bin_conf_matrix_test_1[1,:] )
  tnr_test_1 = bin_conf_matrix_test_1[0,0] / np.sum ( bin_conf_matrix_test_1[0,:] )
  conf_matrices[1] . append ( multi_conf_matrix_test )   # add to the relative container
  tprs[1] . append ( [tpr_test_2, tpr_test_1] )          # add to the relative container
  tnrs[1] . append ( [tnr_test_2, tnr_test_1] )          # add to the relative container

  auc_test_2 = roc_auc_score ( (y_test_comb == 3), y_scores_test_comb[:,1] )                # one-vs-all AUC score (PMBCL class)
  fpr_test_2 , tpr_test_2 , _ = roc_curve ( (y_test_comb == 3), y_scores_test_comb[:,1] )   # one-vs-all ROC curve (PMBCL class)

  if (len(fpr_test_2) == n_roc_points[0]): append_to_roc[0] = True
  if append_to_roc[0] and (len(fpr_test_2) > 10):
    roc_curves[0] . append ( np.c_ [1 - fpr_test_2, tpr_test_2] )   # add to the relative container
    auc_scores[0] . append ( auc_test_2 )                           # add to the relative container
    append_to_roc[0] = False ; n_roc_points[0] = len(fpr_test_2)

  auc_test_1 = roc_auc_score ( (y_test_comb == 2), y_scores_test_comb[:,1] )                # one-vs-all AUC score (GZL class)
  fpr_test_1 , tpr_test_1 , _ = roc_curve ( (y_test_comb == 2), y_scores_test_comb[:,1] )   # one-vs-all ROC curve (GZL class)

  if (len(fpr_test_1) == n_roc_points[1]): append_to_roc[1] = True
  if append_to_roc[1] and (len(fpr_test_1) > 10):
    roc_curves[1] . append ( np.c_ [1 - fpr_test_1, tpr_test_1] )   # add to the relative container
    auc_scores[1] . append ( auc_test_1 )                           # add to the relative container
    append_to_roc[1] = False ; n_roc_points[1] = len(fpr_test_1)

#   +----------------------+
#   |   Plots generation   |
#   +----------------------+

def model_name() -> str:
  if   args.model == "log-reg"   : return "Logistic Regression"
  elif args.model == "lin-svm"   : return "Linear SVM classifier"
  elif args.model == "gaus-proc" : return "Gaussian Process classifier"
  elif args.model == "rnd-frs"   : return "Random Forest classifier"
  elif args.model == "grad-bdt"  : return "Gradient BDT classifier"
  elif args.model == "suv-max"   : return "SUV$_{max}$-based classifier"

plot_conf_matrices ( conf_matrix = np.mean(conf_matrices[0], axis = 0) . astype(np.int32) ,
                     labels = LABELS      ,
                     show_matrix = "both" , 
                     save_figure = True   ,
                     fig_name = f"multi-clf/{args.model}/{args.model}_{args.threshold}_train" )

plot_conf_matrices ( conf_matrix = np.mean(conf_matrices[1], axis = 0) . astype(np.int32) ,
                     labels = LABELS      ,
                     show_matrix = "both" , 
                     save_figure = True   ,
                     fig_name = f"multi-clf/{args.model}/{args.model}_{args.threshold}_test" )

plot_multi_prf_histos ( tpr_scores = ( np.array(tprs[0])[:,0] , np.array(tprs[0])[:,1]    ) ,
                        tnr_scores = ( np.array(tnrs[0])[:,0] , np.array(tnrs[0])[:,1] ) ,
                        bins = 25 ,
                        title = f"Performance of multi-class {model_name()} (on train-set)" ,
                        cls_labels = (LABELS[2], LABELS[1]) ,
                        save_figure = True ,
                        fig_name = f"multi-clf/{args.model}/{args.model}_{args.threshold}_train_prf" )

plot_multi_prf_histos ( tpr_scores = ( np.array(tprs[1])[:,0] , np.array(tprs[1])[:,1]    ) ,
                        tnr_scores = ( np.array(tnrs[1])[:,0] , np.array(tnrs[1])[:,1] ) ,
                        bins = 25 ,
                        title = f"Performance of multi-class {model_name()} (on test-set)" ,
                        cls_labels = (LABELS[2], LABELS[1]) ,
                        save_figure = True ,
                        fig_name = f"multi-clf/{args.model}/{args.model}_{args.threshold}_test_prf" )

#   +-------------------+
#   |   Scores export   |
#   +-------------------+

roc_vars_pmbcl = np.mean ( roc_curves[0], axis = 0 )
auc_vars_pmbcl = np.array ( [ np.percentile ( auc_scores[0], 10, axis = 0 ) , 
                              np.percentile ( auc_scores[0], 32, axis = 0 ) , 
                              np.mean ( auc_scores[0], axis = 0 ) ,
                              np.std  ( auc_scores[0], axis = 0 ) ] )

roc_vars_gzl = np.mean ( roc_curves[1], axis = 0 )
auc_vars_gzl = np.array ( [ np.percentile ( auc_scores[1], 10, axis = 0 ) , 
                            np.percentile ( auc_scores[1], 32, axis = 0 ) , 
                            np.mean ( auc_scores[1], axis = 0 ) ,
                            np.std  ( auc_scores[1], axis = 0 ) ] )

score_dir  = "scores"
score_name = f"{args.model}_{args.threshold}"

filename = f"{score_dir}/multi-clf/{score_name}.npz"
np . savez ( filename  , 
             roc_pmbcl = roc_vars_pmbcl , auc_pmbcl = auc_vars_pmbcl , 
             roc_gzl   = roc_vars_gzl   , auc_gzl   = auc_vars_gzl   )
print (f"Scores correctly exported to {filename}")
