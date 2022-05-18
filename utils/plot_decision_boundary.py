import numpy as np
import matplotlib.pyplot as plt

from datetime        import datetime
from sklearn.metrics import confusion_matrix
from .custom_predictions import custom_predictions

RESOLUTION = 100


def plot_decision_boundary ( y_true    : np.ndarray , 
                             y_scores  : np.ndarray ,
                             X_feat    : np.ndarray ,
                             feat_name : str  ,
                             strategy  : str  = None ,
                             labels    : list = None ,
                             save_figure : bool = False ,
                             fig_name : str = None ) -> None:
  prec70 = np.zeros(2) ; prec80 = np.zeros(2) ; prec90 = np.zeros(2)
  rec70  = np.zeros(2) ; rec80  = np.zeros(2) ; rec90  = np.zeros(2)

  if strategy is not None:
    ## PPV or TPR > 0.7
    if (strategy == "precision") : y_pred70, thr70 = custom_predictions (y_true, y_scores, precision_score = 0.7)
    elif (strategy == "recall")  : y_pred70, thr70 = custom_predictions (y_true, y_scores, recall_score = 0.7)
    cm70 = confusion_matrix (y_true, y_pred70)
    props70 = dict (boxstyle = "round", facecolor = "#bababa", edgecolor = "#404040")

    ## PPV or TPR > 0.8
    if (strategy == "precision") : y_pred80, thr80 = custom_predictions (y_true, y_scores, precision_score = 0.8)
    elif (strategy == "recall")  : y_pred80, thr80 = custom_predictions (y_true, y_scores, recall_score = 0.8)
    cm80 = confusion_matrix (y_true, y_pred80)
    props80 = dict (boxstyle = "round", facecolor = "#bababa", edgecolor = "#404040")

    ## PPV or TPR > 0.9
    if (strategy == "precision") : y_pred90, thr90 = custom_predictions (y_true, y_scores, precision_score = 0.9)
    elif (strategy == "recall")  : y_pred90, thr90 = custom_predictions (y_true, y_scores, recall_score = 0.9)
    cm90 = confusion_matrix (y_true, y_pred90)
    props90 = dict (boxstyle = "round", facecolor = "#bababa", edgecolor = "#404040")

    ## precision/recall computation
    for i in range (2):
      prec70[i] = cm70[i,i] / np.sum ( cm70[:,i] )
      rec70[i]  = cm70[i,i] / np.sum ( cm70[i,:] )
      prec80[i] = cm80[i,i] / np.sum ( cm80[:,i] )
      rec80[i]  = cm80[i,i] / np.sum ( cm80[i,:] )
      prec90[i] = cm90[i,i] / np.sum ( cm90[:,i] )
      rec90[i]  = cm90[i,i] / np.sum ( cm90[i,:] )

    ## box text
    if (strategy == "precision"):
      textstr70 = "\n" . join ( [f"PPV : {prec70[1]:.3f}", f"NPV : {prec70[0]:.3f}"] )
      textstr80 = "\n" . join ( [f"PPV : {prec80[1]:.3f}", f"NPV : {prec80[0]:.3f}"] )
      textstr90 = "\n" . join ( [f"PPV : {prec90[1]:.3f}", f"NPV : {prec90[0]:.3f}"] )
    elif (strategy == "recall"):
      textstr70 = "\n" . join ( [f"TPR : {rec70[1]:.3f}", f"TNR : {rec70[0]:.3f}"] )
      textstr80 = "\n" . join ( [f"TPR : {rec80[1]:.3f}", f"TNR : {rec80[0]:.3f}"] )
      textstr90 = "\n" . join ( [f"TPR : {rec90[1]:.3f}", f"TNR : {rec90[0]:.3f}"] )

  ## true positive and true negative scores
  tp_scores = y_scores[np.nonzero(y_true == True)]
  tn_scores = y_scores[np.nonzero(y_true == False)]

  ## plot limits
  x_min = np.min ( [np.min(tp_scores[:,1]), np.min(tn_scores[:,1])] ) - 0.05
  x_max = np.max ( [np.max(tp_scores[:,1]), np.max(tn_scores[:,1])] ) + 0.05
  y_min = np.min ( X_feat ) - 0.25 * ( np.max(X_feat) - np.min(X_feat) )  
  y_max = np.max ( X_feat ) + 0.25 * ( np.max(X_feat) - np.min(X_feat) )

  ## text positions
  if strategy is not None:
    if (strategy == "precision"):
      x_txt70 = thr70 - 0.5 * (thr70 - x_min)
      x_txt80 = thr80 - 0.5 * (thr80 - thr70)
      x_txt90 = thr90 - 0.5 * (thr90 - thr80)
    elif (strategy == "recall"):
      x_txt70 = thr70 + 0.5 * (x_max - thr70)
      x_txt80 = thr80 + 0.5 * (thr70 - thr80)
      x_txt90 = thr90 + 0.5 * (thr80 - thr90)
    y_txt_top    = y_max - 0.1 * ( y_max - y_min )
    y_txt_bottom = y_min + 0.1 * ( y_max - y_min )

  ## ## figure setup
  fig, ax = plt.subplots (figsize = (8,6), dpi = RESOLUTION)
  ax.set_xlabel ("Predicted PMBCL probability", fontsize = 12)
  ax.set_ylabel (f"{feat_name}", fontsize = 12)

  if strategy is not None:
    ## PPV or TPR > 0.7
    ax.axvspan (x_min, thr70, color = "#f4a582", alpha = 0.15, zorder = 1)
    ax.plot ([thr70,thr70], [y_min,y_max], color = "black", linestyle = "-", zorder = 2)
    ax.axvspan (thr70, x_max, color = "#92c5de", alpha = 0.15, zorder = 1)
    ax.text (x_txt70, y_txt_bottom, textstr70, fontsize = 8, color = "#404040", weight = "bold", ha = "center", va = "center", bbox = props70)

    ## PPV or TPR > 0.8
    ax.axvspan (x_min, thr80, color = "#f4a582", alpha = 0.15, zorder = 1)
    ax.plot ([thr80,thr80], [y_min,y_max], color = "black", linestyle = "--", zorder = 2)
    ax.axvspan (thr80, x_max, color = "#92c5de", alpha = 0.15, zorder = 1)
    ax.text (x_txt80, y_txt_top, textstr80, fontsize = 8, color = "#404040", weight = "bold", ha = "center", va = "center", bbox = props80)

    ## PPV or TPR > 0.9
    ax.axvspan (x_min, thr90, color = "#f4a582", alpha = 0.15, zorder = 1)
    ax.plot ([thr90,thr90], [y_min,y_max], color = "black", linestyle = ":", zorder = 2)
    ax.axvspan (thr90, x_max, color = "#92c5de", alpha = 0.15, zorder = 1)
    ax.text (x_txt90, y_txt_bottom, textstr90, fontsize = 8, color = "#404040", weight = "bold", ha = "center", va = "center", bbox = props90)
    
  ax.scatter (tp_scores[:,1], X_feat[np.nonzero(y_true == True)] , color = "#0571b0", marker = "^", label = f"{labels[1]}", zorder = 4)
  ax.scatter (tn_scores[:,1], X_feat[np.nonzero(y_true == False)], color = "#ca0020", marker = "o", label = f"{labels[0]}", zorder = 3)
  ax.legend (title = "True label", loc = "upper left", fontsize = 10)
  ax.axis ([x_min, x_max, y_min, y_max])

  ## figure name default
  if fig_name is None:
    timestamp = str (datetime.now()) . split (".") [0]
    timestamp = timestamp . replace (" ","_")
    fig_name = "dec_bound_"
    for time, unit in zip ( timestamp.split(":"), ["h","m","s"] ):
      fig_name += time + unit   # YYYY-MM-DD_HHhMMmSSs
  filename = f"docs/img/{fig_name}.png"
  
  plt.tight_layout()
  if save_figure: 
    plt.savefig ( filename, format = "png", dpi = RESOLUTION )
    print (f"Figure correctly exported to {filename}")

  plt.show()
