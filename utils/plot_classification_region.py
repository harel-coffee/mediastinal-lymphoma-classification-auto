import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from .save_figure             import save_figure
from .precision_recall_scores import precision_recall_scores


def plot_classification_region ( y_true : np.ndarray ,
                                 y_scores : np.ndarray ,
                                 X_true_high_rnk = np.ndarray ,
                                 boundary : list = [0.5] ,
                                 labels : list = None ,
                                 high_rnk_feat_name : str = None ,
                                 show_conf_matrix : bool = False ,
                                 fig_name : str = None ) -> None:
  ## Shape check
  if len(y_true) != len(y_scores):
    raise ValueError ( "True labels and predicted scores don't match." )
  if len(y_true) != len(X_true_high_rnk):
    raise ValueError ( "True labels and high ranked feature values don't match." )

  ## Label check
  if labels is not None:
    if len(labels) != len(np.unique(y_true)):
      raise ValueError ( "Labels list length doesn't match with the true number of classes." )
  else:
    labels = np.unique ( y_true )

  ## Boundary check
  if len(labels) != ( len(boundary) + 1 ):
    raise ValueError ( "The number of boundaries passed doesn't match with the number of labels." )

  ## High ranked feature name
  if high_rnk_feat_name is None:
    high_rnk_feat_name = "Most important feature"

  ## Plot limits
  x_min = np.min(y_scores[:,1]) - 0.05
  x_max = np.max(y_scores[:,1]) + 0.05
  y_min = np.min( X_true_high_rnk ) - 0.5
  y_max = np.max( X_true_high_rnk ) + 0.5

  ## Plot classification region
  plt.figure (figsize = (8,6), dpi = 100)
  plt.xlabel ("Predicted PMBCL probability", fontsize = 12)
  plt.ylabel ("{}" . format (high_rnk_feat_name), fontsize = 12)

  ## Classification region
  plt.axvspan (x_min, boundary[0], color = "salmon", alpha = 0.45, zorder = 1)
  plt.plot ([boundary[0],boundary[0]], [y_min,y_max], color = "black", linestyle = "--", zorder = 2)
  if len(labels) == 2:  
    plt.axvspan (boundary[0], x_max, color = "cornflowerblue", alpha = 0.45, zorder = 1)
  elif len(labels) == 3:
    plt.axvspan (boundary[0], boundary[1], color = "limegreen", alpha = 0.45, zorder = 1)
    plt.plot ([boundary[1],boundary[1]], [y_min,y_max], color = "black", linestyle = "--", zorder = 2)
    plt.axvspan (boundary[1], x_max, color = "cornflowerblue", alpha = 0.45, zorder = 1)

  ## Scatter plot
  if len(labels) == 2:
    tn_idx = np.nonzero ( y_true == False )   # true negative
    tp_idx = np.nonzero ( y_true == True )    # true positive
    plt.scatter ( y_scores[:,1][tn_idx], X_true_high_rnk[tn_idx], 
                  color = "red", marker = "o", label = labels[0], zorder = 3 )
    plt.scatter ( y_scores[:,1][tp_idx], X_true_high_rnk[tp_idx], 
                  color = "blue", marker = "^", label = labels[1], zorder = 3 )
  elif len(labels) == 3:
    cls1_idx = np.nonzero ( y_true == np.unique(y_true)[0] )
    cls2_idx = np.nonzero ( y_true == np.unique(y_true)[1] )
    cls3_idx = np.nonzero ( y_true == np.unique(y_true)[2] )
    plt.scatter ( y_scores[:,1][cls1_idx], X_true_high_rnk[cls1_idx], 
                  color = "red", marker = "o", label = labels[0], zorder = 3)
    plt.scatter ( y_scores[:,1][cls2_idx], X_true_high_rnk[cls2_idx], 
                  color = "green", marker = "s", label = labels[1], zorder = 3)
    plt.scatter ( y_scores[:,1][cls3_idx], X_true_high_rnk[cls3_idx], 
                  color = "blue", marker = "^", label = labels[2], zorder = 3)
  plt.legend (title = "True label", loc = "upper left", fontsize = 10)
  plt.axis ([x_min, x_max, y_min, y_max])
  
  ## Save figure
  if fig_name is None:
    timestamp = str (datetime.now()) . split (".") [0]
    timestamp = timestamp . replace (" ","_")
    fig_name = "plot_classification_region_"
    for time, unit in zip ( timestamp.split(":"), ["h","m","s"] ):
      fig_name += time + unit   # YYYY-MM-DD_HHhMMmSSs
  save_figure ( fig_name )

  plt.show()

  if show_conf_matrix:
    if len(labels) == 2:
      pn_idx = np.nonzero ( y_scores[:,1] < boundary[0] )    # predicted negative
      pp_idx = np.nonzero ( y_scores[:,1] >= boundary[0] )   # predicted positive
      y_pred = np.zeros_like ( y_scores[:,1] )
      y_pred[pn_idx] = False
      y_pred[pp_idx] = True
    elif len(labels) == 3:
      p1_idx = np.nonzero ( y_scores[:,1] < boundary[0] )    # predicted class 1
      p3_idx = np.nonzero ( y_scores[:,1] >= boundary[1] )   # predicted class 3
      y_pred = 2 * np.ones_like ( y_scores[:,1] )
      y_pred[p1_idx] = 1.
      y_pred[p3_idx] = 3.

    _ = precision_recall_scores ( y_true , y_pred , labels = labels , 
                                  show_conf_matrix = show_conf_matrix , 
                                  fig_name = f"{fig_name}_conf_matrix" )
