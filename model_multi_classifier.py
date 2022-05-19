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

MODELS = [ "log-reg", "lin-svm", "gaus-proc", "rnd-frs", "grad-bdt" ]

parser = ArgumentParser ( description = "training script" )
parser . add_argument ( "-m" , "--model"     , required = True , choices = MODELS )
parser . add_argument ( "-s" , "--split"     , default  = "50/30/20" )
parser . add_argument ( "-t" , "--threshold" , default  = "rec90" )
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

conf_matrices = [ list() , list() , list() ]   # container for confusion matrices
recalls       = [ list() , list() , list() ]   # container for recalls
precisions    = [ list() , list() , list() ]   # container for precisions
roc_curves    = [ list() , list() ]            # container for ROC curve variables

## initial control values
optimized = False
append_to_roc = [ True , True ]
n_roc_points  = [ -1   , -1   ]

for i in tqdm(range(250)):

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

  #   +------------------+
  #   |   Optuna setup   |
  #   +------------------+

  def optuna_study ( model_name  : str ,
                     storage_dir : str ,
                     objective   : float ,
                     n_trials    : int = 10 ,
                     direction   : str = "minimize" , 
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
                                  direction = direction )

    study . optimize ( objective, n_trials = n_trials )

    return study

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
      n_estims  = trial . suggest_int ( "n_estims"  , 5 , 150 , log = True )
      max_depth = trial . suggest_int ( "max_depth" , 1 , 10  )

      ## model to optimize
      model = RandomForestClassifier ( n_estimators = n_estims   ,
                                       max_depth = max_depth     )

      model.fit (X_trn_res, y_trn_res)
      y_scores = model.predict_proba (X_val)
      return roc_auc_score ( y_val, y_scores[:,1] )   # score to optimize

    if not optimized:
      study = optuna_study ( model_name  = "rnd_forest_clf" ,
                             storage_dir = "./storage"  ,
                             objective = objective ,
                             n_trials  = 50 ,
                             direction = "maximize" ,
                             load_if_exists = False )
      optimized = True

    best_model = RandomForestClassifier ( n_estimators = study.best_params["n_estims"]  ,
                                          max_depth    = study.best_params["max_depth"] )


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
      learn_rate = trial . suggest_float ( "learn_rate" , 5e-2 , 5e-1 , log = True )
      n_estims   = trial . suggest_int   ( "n_estims"   , 5    , 150  , log = True )
      max_depth  = trial . suggest_int   ( "max_depth"  , 1    , 10   )

      ## model to optimize
      model = GradientBoostingClassifier ( learning_rate = learn_rate , 
                                           n_estimators  = n_estims   , 
                                           max_depth     = max_depth  )

      model.fit (X_trn_res, y_trn_res)
      y_scores = model.predict_proba (X_val)
      return roc_auc_score ( y_val, y_scores[:,1] )   # score to optimize

    if not optimized:
      study = optuna_study ( model_name  = "grad_bdt_clf"  ,
                             storage_dir = "./storage" ,
                             objective = objective ,
                             n_trials  = 50 ,
                             direction = "maximize" ,
                             load_if_exists = False )
      optimized = True

    best_model = GradientBoostingClassifier ( learning_rate = study.best_params["learn_rate"] , 
                                              n_estimators  = study.best_params["n_estims"]   , 
                                              max_depth     = study.best_params["max_depth"]  )

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

  ## train/val splitting
  sss = StratifiedShuffleSplit ( n_splits = 1, test_size = val_size )
  for idx_trn, idx_val in sss . split ( X_train, y_train ):
    X_trn , y_trn = X_train[idx_trn] , y_train[idx_trn]
    X_val , y_val = X_train[idx_val] , y_train[idx_val] 

  sm = SMOTE()   # oversampling technique
  X_trn_res, y_trn_res = sm.fit_resample ( X_trn , y_trn )

  ## combine the datasets
  X_trn_comb = np.concatenate ( [ X_trn, X_gz ] )
  y_trn_comb = np.concatenate ( [ np.where(y_trn, 3, 1), y_gz ] )

  X_val_comb = np.concatenate ( [ X_val, X_gz ] )
  y_val_comb = np.concatenate ( [ np.where(y_val, 3, 1), y_gz ] )

  X_test_comb = np.concatenate ( [ X_test, X_gz ] )
  y_test_comb = np.concatenate ( [ np.where(y_test, 3, 1), y_gz ] )

  X_eval_comb = np.concatenate ( [ X_val, X_test, X_gz ] )
  y_eval_comb = np.concatenate ( [ np.where(y_val, 3, 1) , np.where(y_test, 3, 1), y_gz ] )

  ## model training
  best_model . fit (X_trn_res, y_trn_res)

  ## model predictions
  y_scores_trn = best_model.predict_proba ( X_trn )
  _, threshold = custom_predictions ( y_true = y_trn , 
                                      y_scores = y_scores_trn , 
                                      recall_score = rec_score ,
                                      precision_score = prec_score )   

  y_scores_trn_comb = best_model.predict_proba ( X_trn_comb )
  y_pred_trn = multiclass_predictions ( y_true = y_trn_comb ,
                                        y_scores = y_scores_trn_comb ,
                                        boundaries = get_decision_boundaries ( y_scores_trn, threshold, len(y_gz) / len(y_trn_comb) ) )   # pred for the true train-set

  y_scores_val_comb = best_model.predict_proba ( X_val_comb )
  y_pred_val = multiclass_predictions ( y_true = y_val_comb ,
                                        y_scores = y_scores_val_comb ,
                                        boundaries = get_decision_boundaries ( y_scores_trn, threshold, len(y_gz) / len(y_trn_comb) ) )   # pred for the val-set

  y_scores_test_comb = best_model.predict_proba ( X_test_comb )
  y_pred_test = multiclass_predictions ( y_true = y_test_comb ,
                                         y_scores = y_scores_test_comb ,
                                         boundaries = get_decision_boundaries ( y_scores_trn, threshold, len(y_gz) / len(y_trn_comb) ) )   # pred for the test-set

  y_scores_eval_comb = best_model.predict_proba ( X_eval_comb )
  y_pred_eval = multiclass_predictions ( y_true = y_eval_comb ,
                                         y_scores = y_scores_eval_comb ,
                                         boundaries = get_decision_boundaries ( y_scores_trn, threshold, len(y_gz) / len(y_trn_comb) ) )   # pred for the val-set + test-set

  ## model performances
  conf_matrix_trn = confusion_matrix ( y_trn_comb, y_pred_trn )
  recall_2 = conf_matrix_trn[2,2] / np.sum ( conf_matrix_trn[2,:] )
  recall_1 = conf_matrix_trn[1,1] / np.sum ( conf_matrix_trn[1,:] )
  precision_2 = conf_matrix_trn[2,2] / np.sum ( conf_matrix_trn[:,2] )
  precision_1 = conf_matrix_trn[1,1] / np.sum ( conf_matrix_trn[:,1] )
  conf_matrices[0] . append ( conf_matrix_trn )           # add to the relative container
  recalls[0]    . append ( [recall_2, recall_1] )         # add to the relative container
  precisions[0] . append ( [precision_2, precision_1] )   # add to the relative container

  conf_matrix_val = confusion_matrix ( y_val_comb, y_pred_val )
  recall_2 = conf_matrix_val[2,2] / np.sum ( conf_matrix_val[2,:] )
  recall_1 = conf_matrix_val[1,1] / np.sum ( conf_matrix_val[1,:] )
  precision_2 = conf_matrix_val[2,2] / np.sum ( conf_matrix_val[:,2] )
  precision_1 = conf_matrix_val[1,1] / np.sum ( conf_matrix_val[:,1] )
  conf_matrices[1] . append ( conf_matrix_val )           # add to the relative container
  recalls[1]    . append ( [recall_2, recall_1] )         # add to the relative container
  precisions[1] . append ( [precision_2, precision_1] )   # add to the relative container

  conf_matrix_test = confusion_matrix ( y_test_comb, y_pred_test )
  recall_2 = conf_matrix_test[2,2] / np.sum ( conf_matrix_test[2,:] )
  recall_1 = conf_matrix_test[1,1] / np.sum ( conf_matrix_test[1,:] )
  precision_2 = conf_matrix_test[2,2] / np.sum ( conf_matrix_test[:,2] )
  precision_1 = conf_matrix_test[1,1] / np.sum ( conf_matrix_test[:,1] )
  conf_matrices[2] . append ( conf_matrix_test )          # add to the relative container
  recalls[2]    . append ( [recall_2, recall_1] )         # add to the relative container
  precisions[2] . append ( [precision_2, precision_1] )   # add to the relative container

  auc_eval_2 = roc_auc_score ( (y_eval_comb == 3), y_scores_eval_comb[:,1] )                # one-vs-all AUC score (PMBCL class)
  fpr_eval_2 , tpr_eval_2 , _ = roc_curve ( (y_eval_comb == 3), y_scores_eval_comb[:,1] )   # one-vs-all ROC curve (PMBCL class)

  if (len(fpr_eval_2) == n_roc_points[0]): append_to_roc[0] = True
  if append_to_roc[0]:
    roc_curves[0] . append ( np.c_ [1 - fpr_eval_2, tpr_eval_2, auc_eval_2 * np.ones_like(fpr_eval_2)] )   # add to the relative container
    append_to_roc[0] = False ; n_roc_points[0] = len(fpr_eval_2)

  auc_eval_1 = roc_auc_score ( (y_eval_comb == 2), y_scores_eval_comb[:,1] )                # one-vs-all AUC score (GZL class)
  fpr_eval_1 , tpr_eval_1 , _ = roc_curve ( (y_eval_comb == 2), y_scores_eval_comb[:,1] )   # one-vs-all ROC curve (GZL class)

  if (len(fpr_eval_1) == n_roc_points[1]): append_to_roc[1] = True
  if append_to_roc[1]:
    roc_curves[1] . append ( np.c_ [1 - fpr_eval_1, tpr_eval_1, auc_eval_1 * np.ones_like(fpr_eval_1)] )   # add to the relative container
    append_to_roc[1] = False ; n_roc_points[1] = len(fpr_eval_1)

#   +----------------------+
#   |   Plots generation   |
#   +----------------------+

def model_name() -> str:
  if   args.model == "log-reg"   : return "Logistic Regression"
  elif args.model == "lin-svm"   : return "Linear SVM classifier"
  elif args.model == "gaus-proc" : return "Gaussian Process classifier"
  elif args.model == "rnd-frs"   : return "Random Forest classifier"
  elif args.model == "grad-bdt"  : return "Gradient BDT classifier"

plot_conf_matrices ( conf_matrix = np.mean(conf_matrices[0], axis = 0) . astype(np.int32) ,
                     labels = LABELS      ,
                     show_matrix = "both" , 
                     save_figure = True   ,
                     fig_name = f"multi-clf/{args.model}/{args.model}_{args.threshold}_train" )

plot_conf_matrices ( conf_matrix = np.mean(conf_matrices[1], axis = 0) . astype(np.int32) ,
                     labels = LABELS      ,
                     show_matrix = "both" , 
                     save_figure = True   ,
                     fig_name = f"multi-clf/{args.model}/{args.model}_{args.threshold}_val" )

plot_conf_matrices ( conf_matrix = np.mean(conf_matrices[2], axis = 0) . astype(np.int32) ,
                     labels = LABELS      ,
                     show_matrix = "both" , 
                     save_figure = True   ,
                     fig_name = f"multi-clf/{args.model}/{args.model}_{args.threshold}_test" )

plot_multi_prf_histos ( rec_scores  = ( np.array(recalls[0])[:,0]    , np.array(recalls[0])[:,1]    ) ,
                        prec_scores = ( np.array(precisions[0])[:,0] , np.array(precisions[0])[:,1] ) ,
                        bins = 25 ,
                        title = f"Performance of multi-class {model_name()} (on train-set)" ,
                        cls_labels = (LABELS[2], LABELS[1]) ,
                        save_figure = True ,
                        fig_name = f"multi-clf/{args.model}/{args.model}_{args.threshold}_train_prf" )

plot_multi_prf_histos ( rec_scores  = ( np.array(recalls[1])[:,0]    , np.array(recalls[1])[:,1]    ) ,
                        prec_scores = ( np.array(precisions[1])[:,0] , np.array(precisions[1])[:,1] ) ,
                        bins = 25 ,
                        title = f"Performance of multi-class {model_name()} (on val-set)" ,
                        cls_labels = (LABELS[2], LABELS[1]) ,
                        save_figure = True ,
                        fig_name = f"multi-clf/{args.model}/{args.model}_{args.threshold}_val_prf" )

plot_multi_prf_histos ( rec_scores  = ( np.array(recalls[2])[:,0]    , np.array(recalls[2])[:,1]    ) ,
                        prec_scores = ( np.array(precisions[2])[:,0] , np.array(precisions[2])[:,1] ) ,
                        bins = 25 ,
                        title = f"Performance of multi-class {model_name()} (on test-set)" ,
                        cls_labels = (LABELS[2], LABELS[1]) ,
                        save_figure = True ,
                        fig_name = f"multi-clf/{args.model}/{args.model}_{args.threshold}_test_prf" )

#   +-------------------+
#   |   Scores export   |
#   +-------------------+

roc_vars_lbl3 = np.c_ [ np.mean(roc_curves[0], axis = 0) , np.std(roc_curves[0], axis = 0)[:,2] ]
roc_vars_lbl2 = np.c_ [ np.mean(roc_curves[1], axis = 0) , np.std(roc_curves[1], axis = 0)[:,2] ]

score_dir  = "scores"
score_name = f"{args.model}_{args.threshold}"

filename = f"{score_dir}/multi-clf/{score_name}.npz"
np . savez ( filename, roc_vars_lbl3 = roc_vars_lbl3, roc_vars_lbl2 = roc_vars_lbl2 )
print (f"Scores correctly exported to {filename}")
