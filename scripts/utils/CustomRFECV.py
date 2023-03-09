import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling  import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from multiprocessing import Pool


class CustomRFECV:
  def __init__ ( self , 
                 estimator , 
                 cv = 2 , 
                 scoring = "roc_auc" ,
                 iterations = 1 ,
                 n_jobs = None ) -> None:
    ## data-types control
    if not isinstance (cv, int):
      if isinstance (cv, float): cv = int(cv)
      else: TypeError ("The number f K-Folds should be passed as integer.")
    if not isinstance (scoring, str):
      raise TypeError ("The scoring strategy should be passed as string.")
    if not isinstance (iterations, int):
      if isinstance (iterations, float): iterations = int(iterations)
      else: TypeError ("The number of training iterations should be passed as integer.")
    if n_jobs is not None:
      if not isinstance (n_jobs, int):
        if isinstance (n_jobs, float): n_jobs = int(n_jobs)
        else: TypeError ("The number of CPU cores should be passed as integer.")

    ## data-values control
    if cv < 2:
      raise ValueError ("The number of K-Folds should be greater or equal to 2.")
    if scoring not in ["accuracy", "precision", "recall", "roc_auc"]:
      raise ValueError ("The scoring strategy should be chosen within ['accuracy', 'precision', 'recall', 'roc_auc'].")
    if iterations < 1:
      raise ValueError ("The number of training iterations should be greater or equal to 1.")
    if n_jobs is not None:
      if n_jobs < 1:
        raise ValueError ("The number of training iterations should be greater or equal to 1.")
    
    self._sm  = SMOTE()
    self._skf = StratifiedKFold ( n_splits = cv )
    self._estimator  = estimator
    self._scoring    = scoring
    self._iterations = iterations
    self._n_jobs = n_jobs

  def fit ( self, X, y ) -> None:
    discarded = list()
    best_mask = np.array ( [True for i_var in range(X.shape[1])] )

    while np.count_nonzero (best_mask) > 1:

      scores = list()
      for i_var in range ( len(best_mask) ):
        if best_mask[i_var] == False:
          scores . append (2)
        else:
          mask = best_mask.copy()
          mask[i_var] = False
          with Pool(self._n_jobs) as p:
            p_out = p.starmap ( self._cross_val_mean, [ (X.T[mask].T, y) for _ in range(self._iterations) ] )
          tmp_score = np.mean (p_out)
          scores . append (tmp_score)

      ranks = np.argsort ( scores )
      discarded . append ( ranks[0] )
      best_mask [ranks[0]] = False

    discarded . append ( int ( np.arange (len(best_mask)) [best_mask] ) )
    self._ranking = np.flip ( discarded )

  def _cross_val_mean ( self, X, y ) -> np.ndarray:
    cv_scores = list()
    for train_id, test_id in self._skf . split (X, y):
      X_train, X_test = X[train_id], X[test_id]
      y_train, y_test = y[train_id], y[test_id]
      X_train_res, y_train_res = self._sm . fit_resample ( X_train, y_train )
      self._estimator . fit ( X_train_res, y_train_res )
      y_scores = self._estimator . predict_proba ( X_test ) [:,1]
      if   self._scoring == "accuracy"  : cv_scores . append ( accuracy_score  ( y_test , (y_scores > 0.5) ) )
      elif self._scoring == "precision" : cv_scores . append ( precision_score ( y_test , (y_scores > 0.5) ) )
      elif self._scoring == "recall"    : cv_scores . append ( recall_score    ( y_test , (y_scores > 0.5) ) )
      elif self._scoring == "roc_auc"   : cv_scores . append ( roc_auc_score   ( y_test , y_scores ) )
    return np.mean ( cv_scores )

  @property
  def ranking (self) -> np.ndarray:
    return self._ranking
