import os
import pickle
import numpy as np

from tqdm import tqdm
from time import time
from datetime import datetime
from argparse import ArgumentParser

import optuna
optuna.logging.set_verbosity ( optuna.logging.ERROR )   # silence Optuna during trials study

from sklearn.model_selection   import StratifiedShuffleSplit
from sklearn.preprocessing     import MinMaxScaler
from imblearn.over_sampling    import SMOTE
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import SVC
from sklearn.gaussian_process  import GaussianProcessClassifier
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics           import roc_auc_score, confusion_matrix, roc_curve

from utils import custom_predictions, plot_conf_matrices, plot_prf_histos, plot_feat_importance

LABELS = ["cHL", "PMBCL"]
VERSIONING = False

#   +-------------------+
#   |   Options setup   |
#   +-------------------+

MODELS = [ "log-reg", "lin-svm", "gaus-proc", "rnd-frs", "grad-bdt" ]

parser = ArgumentParser ( description = "training script" )
parser . add_argument ( "-m" , "--model"     , required = True , choices = MODELS )
parser . add_argument ( "-s" , "--split"     , default  = "60/20/20" )
parser . add_argument ( "-t" , "--threshold" , default  = "rec80" )
# parser . add_argument ( "-v" , "--version"   , required = True )
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

id = data.query("lymphoma_type != 2")["ID"]   . to_numpy() . flatten()
X  = data.query("lymphoma_type != 2")[X_cols] . to_numpy()
y  = data.query("lymphoma_type != 2")[y_cols] . to_numpy() . flatten()
y  = ( y == 3 )   # PMBCL/cHL classification

#   +------------------------+
#   |   Sub-sample studies   |
#   +------------------------+

conf_matrices = [ list() , list() , list() ]   # container for confusion matrices
tprs = [ list() , list() , list() ]            # container for TPRs
tnrs = [ list() , list() , list() ]            # container for TNRs
roc_curves  = list()                  # container for ROC curve variables
importances = list()                  # container for feature importances

## initial control values
optimized = False
append_to_roc = True
n_roc_points  = -1

for i in tqdm(range(100)):

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

  scaler  = MinMaxScaler()
  X_train = scaler . fit_transform ( X_train )
  X_test  = scaler . transform ( X_test )

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

    # df = study . trials_dataframe ( attrs = ("params", "value") )
    # df_head = df . sort_values ( by = "value", ascending = False ) . head()
    # print ( df_head )

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

    # df = study . trials_dataframe ( attrs = ("params", "value") )
    # df_head = df . sort_values ( by = "value", ascending = False ) . head()
    # print ( df_head )

    best_model = GradientBoostingClassifier ( learning_rate = study.best_params["learn_rate"] , 
                                              n_estimators  = study.best_params["n_estims"]   , 
                                              max_depth     = study.best_params["max_depth"]  )

  #   +-----------------------------------------+
  #   |   Model performance on train/test set   |
  #   +-----------------------------------------+

  ## train/val splitting
  sss = StratifiedShuffleSplit ( n_splits = 1, test_size = val_size )
  for idx_train, idx_val in sss . split ( X_train, y_train ):
    X_trn , y_trn = X_train[idx_train] , y_train[idx_train]
    X_val , y_val = X_train[idx_val]   , y_train[idx_val] 

  sm = SMOTE()   # oversampling technique
  X_trn_res, y_trn_res = sm.fit_resample ( X_trn , y_trn )

  ## model training
  best_model . fit (X_trn_res, y_trn_res)

  ## model predictions
  y_scores_trn = best_model.predict_proba ( X_trn )
  y_pred_trn , threshold = custom_predictions ( y_true = y_trn , 
                                                y_scores = y_scores_trn , 
                                                recall_score = rec_score ,
                                                precision_score = prec_score )   # pred for the true train-set

  y_scores_val = best_model.predict_proba ( X_val )
  y_pred_val = ( y_scores_val[:,1] >= threshold )   # pred for the val-set

  y_scores_test = best_model.predict_proba ( X_test )
  y_pred_test = ( y_scores_test[:,1] >= threshold )   # pred for the test-set

  y_scores_eval = best_model.predict_proba ( np.concatenate ([X_val, X_test]) )
  y_pred_eval = ( y_scores_eval[:,1] >= threshold )   # pred for the val-set + test-set

  ## model performances
  conf_matrix_trn = confusion_matrix ( y_trn, y_pred_trn )
  tpr_trn = conf_matrix_trn[1,1] / np.sum ( conf_matrix_trn[1,:] )
  tnr_trn = conf_matrix_trn[0,0] / np.sum ( conf_matrix_trn[0,:] )
  conf_matrices[0] . append (conf_matrix_trn)   # add to the relative container
  tprs[0] . append (tpr_trn)                    # add to the relative container
  tnrs[0] . append (tnr_trn)                    # add to the relative container

  conf_matrix_val = confusion_matrix ( y_val, y_pred_val )
  tpr_val = conf_matrix_val[1,1] / np.sum ( conf_matrix_val[1,:] )
  tnr_val = conf_matrix_val[0,0] / np.sum ( conf_matrix_val[0,:] )
  conf_matrices[1] . append (conf_matrix_val)   # add to the relative container
  tprs[1] . append (tpr_val)                    # add to the relative container
  tnrs[1] . append (tnr_val)                    # add to the relative container

  conf_matrix_test = confusion_matrix ( y_test, y_pred_test )
  tpr_test = conf_matrix_test[1,1] / np.sum ( conf_matrix_test[1,:] )
  tnr_test = conf_matrix_test[0,0] / np.sum ( conf_matrix_test[0,:] )
  conf_matrices[2] . append (conf_matrix_test)   # add to the relative container
  tprs[2] . append (tpr_test)                    # add to the relative container
  tnrs[2] . append (tnr_test)                    # add to the relative container

  auc_eval = roc_auc_score ( np.concatenate([y_val, y_test]), y_scores_eval[:,1] )
  fpr_eval , tpr_eval , _ = roc_curve ( np.concatenate([y_val, y_test]), y_scores_eval[:,1] )

  if (len(fpr_eval) == n_roc_points): append_to_roc = True

  if append_to_roc:
    roc_curves . append ( np.c_ [1 - fpr_eval, tpr_eval, auc_eval * np.ones_like(fpr_eval)] )
    append_to_roc = False ; n_roc_points = len(fpr_eval)

  ## feature importances
  try:
    selector = RFECV (best_model, step = 1, cv = 3)
    selector . fit (X_train, y_train)
    importances . append ( selector.cv_results_["mean_test_score"] )
  except:
    importances = None

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
                     fig_name = f"bin_clf/{args.model}/{args.model}_{args.threshold}_train" )

plot_conf_matrices ( conf_matrix = np.mean(conf_matrices[1], axis = 0) . astype(np.int32) ,
                     labels = LABELS      ,
                     show_matrix = "both" , 
                     save_figure = True   ,
                     fig_name = f"bin_clf/{args.model}/{args.model}_{args.threshold}_val" )

plot_conf_matrices ( conf_matrix = np.mean(conf_matrices[2], axis = 0) . astype(np.int32) ,
                     labels = LABELS      ,
                     show_matrix = "both" , 
                     save_figure = True   ,
                     fig_name = f"bin_clf/{args.model}/{args.model}_{args.threshold}_test" )

plot_prf_histos ( tpr_scores = np.array(tprs[0]) ,
                  tnr_scores = np.array(tnrs[0]) ,
                  bins = 20 ,
                  title = f"Performance of {model_name()} (on train-set)" ,
                  save_figure = True ,
                  fig_name = f"bin_clf/{args.model}/{args.model}_{args.threshold}_train_prf" )

plot_prf_histos ( tpr_scores = np.array(tprs[1]) ,
                  tnr_scores = np.array(tnrs[1]) ,
                  bins = 20 ,
                  title = f"Performance of {model_name()} (on val-set)" ,
                  save_figure = True ,
                  fig_name = f"bin_clf/{args.model}/{args.model}_{args.threshold}_val_prf" )

plot_prf_histos ( tpr_scores = np.array(tprs[2]) ,
                  tnr_scores = np.array(tnrs[2]) ,
                  bins = 20 ,
                  title = f"Performance of {model_name()} (on test-set)" ,
                  save_figure = True ,
                  fig_name = f"bin_clf/{args.model}/{args.model}_{args.threshold}_test_prf" )

if importances:
  feat_names = [ "SUV_midpoint" , "SUV_mean" , "TLG (mL)" , "SUV_skewness" , "SUV_kurtosis" , "GLCM_homogeneity" , 
                 "GLCM_entropy ($\log_{10}$)" , "GLRLM_SRE" , "GLRLM_LRE" , "GLZLM_LGZE" , "GLZLM_HGZE" ]
  
  plot_feat_importance ( feat_scores = np.mean(importances, axis = 0) ,
                         feat_errors = np.std(importances, axis = 0)  ,
                         feat_names  = feat_names ,
                         title = f"Feature importances for {model_name()}" ,
                         save_figure = True ,
                         fig_name = f"{args.model}/{args.model}_{args.threshold}_feat_imp" )
else:
  print ("Warning! The model selected doesn't allow to study the feature importance.")

#   +-------------------+
#   |   Scores export   |
#   +-------------------+

roc_vars = np.c_ [ np.mean(roc_curves, axis = 0) , np.std(roc_curves, axis = 0)[:,2] ]

score_dir  = "scores"
score_name = f"{args.model}_{args.threshold}"

if VERSIONING:
  score_name += version

filename = f"{score_dir}/{score_name}.npz"
np . savez ( filename, roc_vars = roc_vars )
print (f"Scores correctly exported to {filename}")
