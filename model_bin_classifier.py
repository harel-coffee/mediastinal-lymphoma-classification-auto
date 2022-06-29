#  Run with:
#    python model_bin_classifier.py -m log-reg -s 60/40 -t rec90


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
from sklearn.metrics           import roc_auc_score, confusion_matrix, roc_curve

from utils import custom_predictions, CustomRFECV, plot_conf_matrices, plot_bin_prf_histos, plot_feat_importance

LABELS = ["cHL", "PMBCL"]
DEBUG  = False

#   +-------------------+
#   |   Options setup   |
#   +-------------------+

MODELS = [ "log-reg", "lin-svm", "gaus-proc", "rnd-frs", "grad-bdt", "suv-max" ]

parser = ArgumentParser ( description = "training script" )
parser . add_argument ( "-m" , "--model"     , required = True , choices = MODELS )
parser . add_argument ( "-s" , "--split"     , default  = "60/40" )
parser . add_argument ( "-t" , "--threshold" , default  = "rec90" )
parser . add_argument ( "-f" , "--feat_rank" , default  = "no" , choices = ["yes", "no"] )
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

feat_ranking = ( args.feat_rank == "yes" )

if not feat_ranking:
  NUM_LOOPS  = 300 if not DEBUG else 10
  ITERATIONS = None 
else:
  NUM_LOOPS  = 100 if not DEBUG else 10
  ITERATIONS = max ( 10, 2 * NUM_JOBS ) if not DEBUG else 2

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

id = data.query("lymphoma_type != 2")["ID"]   . to_numpy() . flatten()
X  = data.query("lymphoma_type != 2")[X_cols] . to_numpy()
y  = data.query("lymphoma_type != 2")[y_cols] . to_numpy() . flatten()
y  = ( y == 3 )   # PMBCL/cHL classification

#   +------------------------+
#   |   Sub-sample studies   |
#   +------------------------+

conf_matrices = [ list() , list() ]   # container for confusion matrices
tprs = [ list() , list() ]            # container for TPRs
tnrs = [ list() , list() ]            # container for TNRs
roc_curves = list()                   # container for ROC curve variables
auc_scores = list()                   # container for AUC score values
rankings   = list()                   # container for feature rankings

## initial control values
optimized = False
append_to_roc = True
n_roc_points  = -1

for i in tqdm(range(NUM_LOOPS)):

  #   +--------------------------+
  #   |   Train/test splitting   |
  #   +--------------------------+

  sss = StratifiedShuffleSplit ( n_splits = 1, test_size = test_size )
  for idx_train, idx_test in sss . split ( X, y ):
    X_train , y_train , id_train = X[idx_train] , y[idx_train] , id[idx_train]
    X_test  , y_test  , id_test  = X[idx_test]  , y[idx_test]  , id[idx_test]

  #   +------------------------+
  #   |   Data preprocessing   |
  #   +------------------------+

  scaler_train = MinMaxScaler()
  X_train = scaler_train . fit_transform (X_train)

  scaler_test = MinMaxScaler()
  X_test = scaler_test . fit_transform (X_test)

  #   +------------------------------+
  #   |   Hyperparams optimization   |
  #   +------------------------------+

  ## LOGISTIC REGRESSION
  if args.model == "log-reg":
    best_model = LogisticRegression()

  ## LINEAR SVM
  elif args.model == "lin-svm":
    best_model = SVC ( kernel = "linear", probability = True )

  ## GAUSSIAN PROCESS
  elif args.model == "gaus-proc":
    best_model = GaussianProcessClassifier()

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
                                          max_features     = None )

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

  #   +-----------------------------------------+
  #   |   Model performance on train/test set   |
  #   +-----------------------------------------+

  if best_model is not None:
    sm = SMOTE()   # oversampling technique
    X_train_res, y_train_res = sm.fit_resample ( X_train , y_train )

    ## model training
    best_model . fit (X_train_res, y_train_res)

  ## model predictions
  y_scores_train = best_model.predict_proba ( X_train ) if (best_model is not None) else np.c_ [ 1 - X_train[:,0] , X_train[:,0] ]
  y_pred_train , threshold = custom_predictions ( y_true = y_train , 
                                                  y_scores = y_scores_train , 
                                                  recall_score = rec_score  ,
                                                  precision_score = prec_score )   # pred for the true train-set

  y_scores_test = best_model.predict_proba ( X_test ) if (best_model is not None) else np.c_ [ 1 - X_test[:,0] , X_test[:,0] ]
  y_pred_test = ( y_scores_test[:,1] >= threshold )   # pred for the test-set

  ## model performances
  conf_matrix_train = confusion_matrix ( y_train, y_pred_train )
  tpr_train = conf_matrix_train[1,1] / np.sum ( conf_matrix_train[1,:] )
  tnr_train = conf_matrix_train[0,0] / np.sum ( conf_matrix_train[0,:] )
  conf_matrices[0] . append (conf_matrix_train)   # add to the relative container
  tprs[0] . append (tpr_train)                    # add to the relative container
  tnrs[0] . append (tnr_train)                    # add to the relative container

  conf_matrix_test = confusion_matrix ( y_test, y_pred_test )
  tpr_test = conf_matrix_test[1,1] / np.sum ( conf_matrix_test[1,:] )
  tnr_test = conf_matrix_test[0,0] / np.sum ( conf_matrix_test[0,:] )
  conf_matrices[1] . append (conf_matrix_test)   # add to the relative container
  tprs[1] . append (tpr_test)                    # add to the relative container
  tnrs[1] . append (tnr_test)                    # add to the relative container

  auc_test = roc_auc_score ( y_test, y_scores_test[:,1] )
  fpr_test , tpr_test , _ = roc_curve ( y_test, y_scores_test[:,1] )

  if (len(fpr_test) == n_roc_points): append_to_roc = True

  if append_to_roc and (len(fpr_test) > 10):
    roc_curves . append ( np.c_ [1 - fpr_test, tpr_test] )
    auc_scores . append ( auc_test )
    append_to_roc = False ; n_roc_points = len(fpr_test)

  ## feature rankings
  if feat_ranking and (best_model is not None):
    selector = CustomRFECV (best_model, cv = 3, scoring = "roc_auc", iterations = ITERATIONS)
    selector . fit (X_train, y_train)
    rankings . append (selector.ranking)

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
                     fig_name = f"bin-clf/{args.model}/{args.model}_{args.threshold}_train" )

plot_conf_matrices ( conf_matrix = np.mean(conf_matrices[1], axis = 0) . astype(np.int32) ,
                     labels = LABELS      ,
                     show_matrix = "both" , 
                     save_figure = True   ,
                     fig_name = f"bin-clf/{args.model}/{args.model}_{args.threshold}_test" )

plot_bin_prf_histos ( tpr_scores = np.array(tprs[0]) ,
                      tnr_scores = np.array(tnrs[0]) ,
                      bins = 25 ,
                      title = f"Performance of {model_name()} (on train-set)" ,
                      save_figure = True ,
                      fig_name = f"bin-clf/{args.model}/{args.model}_{args.threshold}_train_prf" )

plot_bin_prf_histos ( tpr_scores = np.array(tprs[1]) ,
                      tnr_scores = np.array(tnrs[1]) ,
                      bins = 25 ,
                      title = f"Performance of {model_name()} (on test-set)" ,
                      save_figure = True ,
                      fig_name = f"bin-clf/{args.model}/{args.model}_{args.threshold}_test_prf" )

if feat_ranking and (best_model is not None):
  feat_names = [ "SUV_max" , "SUV_mean" , "TLG (mL)" , "SUV_skewness" , "SUV_kurtosis" , "GLCM_homogeneity" , 
                 "GLCM_entropy ($\log_{10}$)" , "GLRLM_SRE" , "GLRLM_LRE" , "GLZLM_LGZE" , "GLZLM_HGZE" ]
  
  plot_feat_importance ( feat_ranks = np.array (rankings) ,
                         feat_names = feat_names ,
                         save_figure = True ,
                         fig_name = f"bin-clf/{args.model}/{args.model}_{args.threshold}_feat_imp" )

#   +-------------------+
#   |   Scores export   |
#   +-------------------+

roc_vars = np.mean ( roc_curves, axis = 0 )
auc_vars = np.array ( [ np.percentile ( auc_scores, 10, axis = 0 ) , 
                        np.percentile ( auc_scores, 32, axis = 0 ) , 
                        np.mean ( auc_scores, axis = 0 ) ,
                        np.std  ( auc_scores, axis = 0 ) ] )

score_dir  = "scores"
score_name = f"{args.model}_{args.threshold}"

filename = f"{score_dir}/bin-clf/{score_name}.npz"
np . savez ( filename, roc = roc_vars, auc = auc_vars )
print (f"Scores correctly exported to {filename}")
