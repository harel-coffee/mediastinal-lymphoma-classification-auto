import numpy as np
import matplotlib.pyplot as plt

from sklearn.base              import BaseEstimator
from sklearn.model_selection   import cross_val_predict
from sklearn.feature_selection import RFECV

from .custom_predictions       import custom_predictions
from .precision_recall_scores  import precision_recall_scores


def plot_decision_boundary ( model : BaseEstimator , 
                             X_true : np.ndarray , 
                             y_true : np.ndarray ,
                             labels : list = [0,1] ,
                             strategy : str = None ,
                             feature_names : list = None ) -> None:
  ## Shape check
  if X_true.shape[0] != len(y_true):
    raise ValueError ( "`X_true` and `y_true` lengths don't match." )

  ## Label check
  if len(labels) != 2:
    raise ValueError ( f"A binary problem has two labels, instead {len(labels)} passed." )

  ## Predicted scores
  scores = cross_val_predict ( model, X_true, y_true, cv = 3, method = "predict_proba" )
  tp_scores = scores[np.nonzero(y_true == True)]
  tn_scores = scores[np.nonzero(y_true == False)]

  ## Feature ranking
  selector = RFECV ( model, step = 1, cv = 3 )
  selector . fit ( X_true, y_true )
  ranks = selector . ranking_
  high_rank_idx = np.argmin(ranks)

  ## High ranked feature
  tp_high_rnk_feat = X_true[:,high_rank_idx][np.nonzero(y_true == True)]
  tn_high_rnk_feat = X_true[:,high_rank_idx][np.nonzero(y_true == False)]
  if feature_names is not None:
    if X_true.shape[1] != len(feature_names):
      raise ValueError ( f"The feature space has {X_true.shape[1]} dimensions, "
                         f"but the feature names passed are {len(feature_names)}." )
    else:
      high_rnk_feat = feature_names[high_rank_idx]
  else:
    high_rnk_feat = "Most important feature"

  if strategy is not None:
    ## PPV or TPR > 0.7
    if strategy == "precision":
      pred70, thr70 = custom_predictions (y_true, scores, precision_score = 0.7)
      prec70, _ = precision_recall_scores (y_true, pred70)
      textstr70 = "\n" . join ( [f"PPV : {prec70[1]:.3f}", f"NPV : {prec70[0]:.3f}"] )
    elif strategy == "recall":
      pred70, thr70 = custom_predictions (y_true, scores, recall_score = 0.7)
      _, rec70 = precision_recall_scores (y_true, pred70)
      textstr70 = "\n" . join ( [f"TPR : {rec70[1]:.3f}", f"TNR : {rec70[0]:.3f}"] )
    props70 = dict (boxstyle = "round", facecolor = "wheat")

    ## PPV or TPR > 0.8
    if strategy == "precision":
      pred80, thr80 = custom_predictions (y_true, scores, precision_score = 0.8)
      prec80, _ = precision_recall_scores (y_true, pred80)
      textstr80 = "\n" . join ( [f"PPV : {prec80[1]:.3f}", f"NPV : {prec80[0]:.3f}"] )
    elif strategy == "recall":
      pred80, thr80 = custom_predictions (y_true, scores, recall_score = 0.8)
      _, rec80 = precision_recall_scores (y_true, pred80)
      textstr80 = "\n" . join ( [f"TPR : {rec80[1]:.3f}", f"TNR : {rec80[0]:.3f}"] )
    props80 = dict (boxstyle = "round", facecolor = "wheat")

    ## PPV or TPR > 0.9
    if strategy == "precision":
      pred90, thr90 = custom_predictions (y_true, scores, precision_score = 0.9)
      prec90, _ = precision_recall_scores (y_true, pred90)
      textstr90 = "\n" . join ( [f"PPV : {prec90[1]:.3f}", f"NPV : {prec90[0]:.3f}"] )
    elif strategy == "recall":
      pred90, thr90 = custom_predictions (y_true, scores, recall_score = 0.9)
      _, rec90 = precision_recall_scores (y_true, pred90)
      textstr90 = "\n" . join ( [f"TPR : {rec90[1]:.3f}", f"TNR : {rec90[0]:.3f}"] )
    props90 = dict (boxstyle = "round", facecolor = "wheat")

  ## Plot limits
  x_min = np.min ( [np.min(tp_scores[:,1]), np.min(tn_scores[:,1])] ) - 0.05
  x_max = np.max ( [np.max(tp_scores[:,1]), np.max(tn_scores[:,1])] ) + 0.05
  y_min = np.min ( X_true[:,high_rank_idx] ) - 0.5  
  y_max = np.max ( X_true[:,high_rank_idx] ) + 0.5

  ## Text positions
  if strategy is not None:
    if strategy == "precision":
      x_txt70 = thr70 - 0.5 * (thr70 - x_min)
      x_txt80 = thr80 - 0.5 * (thr80 - thr70)
      x_txt90 = thr90 - 0.5 * (thr90 - thr80)
    elif strategy == "recall":
      x_txt70 = thr70 + 0.5 * (x_max - thr70)
      x_txt80 = thr80 + 0.5 * (thr70 - thr80)
      x_txt90 = thr90 + 0.5 * (thr80 - thr90)
    y_txt = y_min + 0.25

  ## Plot classification results
  plt.figure (figsize = (8,6), dpi = 100)
  plt.xlabel ("Predicted MDLCBL probability", fontsize = 12)
  plt.ylabel ("{}" . format (high_rnk_feat), fontsize = 12)

  if strategy is not None:
    ## PPV or TPR > 0.7
    plt.axvspan (x_min, thr70, color = "salmon", alpha = 0.15, zorder = 1)
    plt.plot ([thr70,thr70], [y_min,y_max], color = "black", linestyle = "-", zorder = 2)
    plt.axvspan (thr70, x_max, color = "cornflowerblue", alpha = 0.15, zorder = 1)
    plt.text (x_txt70, y_txt, textstr70, fontsize = 8, weight = "bold", ha = "center", va = 'center', bbox = props70)

    ## PPV or TPR > 0.8
    plt.axvspan (x_min, thr80, color = "salmon", alpha = 0.15, zorder = 1)
    plt.plot ([thr80,thr80], [y_min,y_max], color = "black", linestyle = "--", zorder = 2)
    plt.axvspan (thr80, x_max, color = "cornflowerblue", alpha = 0.15, zorder = 1)
    plt.text (x_txt80, y_txt, textstr80, fontsize = 8, weight = "bold", ha = "center", va = 'center', bbox = props80)

    ## PPV or TPR > 0.9
    plt.axvspan (x_min, thr90, color = "salmon", alpha = 0.15, zorder = 1)
    plt.plot ([thr90,thr90], [y_min,y_max], color = "black", linestyle = ":", zorder = 2)
    plt.axvspan (thr90, x_max, color = "cornflowerblue", alpha = 0.15, zorder = 1)
    plt.text (x_txt90, y_txt, textstr90, fontsize = 8, weight = "bold", ha = "center", va = 'center', bbox = props90)
    
  plt.scatter (tp_scores[:,1], tp_high_rnk_feat, color = "blue", marker = "^", label = f"{labels[1]}", zorder = 3)
  plt.scatter (tn_scores[:,1], tn_high_rnk_feat, color = "red", marker = "o", label = f"{labels[0]}", zorder = 3)
  plt.legend (title = "True label", loc = "upper left", fontsize = 10)
  plt.axis ([x_min, x_max, y_min, y_max])
  plt.show()
