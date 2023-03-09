import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

RESOLUTION = 300


def plot_clf_region ( y_true    : np.ndarray , 
                      y_scores  : np.ndarray ,
                      X_feat    : np.ndarray ,
                      feat_name : str  ,
                      boundary  : list = None ,
                      labels    : list = None ,
                      save_figure : bool = False ,
                      fig_name : str = None ) -> None:
  ## figure setup
  fig, ax = plt.subplots (figsize = (8,6), dpi = RESOLUTION)
  ax.set_xlabel ("Predicted PMBCL probability", fontsize = 12)
  ax.set_ylabel (f"{feat_name}", fontsize = 12)

  ## plot limits
  x_min = np.min(y_scores[:,1]) - 0.05
  x_max = np.max(y_scores[:,1]) + 0.05
  y_min = np.min ( X_feat ) - 0.25 * ( np.max(X_feat) - np.min(X_feat) )  
  y_max = np.max ( X_feat ) + 0.25 * ( np.max(X_feat) - np.min(X_feat) )

  ## default value for boundary
  if boundary is None: boundary = [ 0.5 * (x_min + x_max) ]

  ## classification region
  ax.axvspan ( x_min, boundary[0], color = "#f4a582", alpha = 0.45, zorder = 1 )
  ax.plot ( [boundary[0], boundary[0]], [y_min, y_max], color = "black", linestyle = "--", zorder = 2 )
  if len(labels) == 2:  
    ax.axvspan ( boundary[0], x_max, color = "#92c5de", alpha = 0.45, zorder = 1 )
  elif len(labels) == 3:
    ax.axvspan ( boundary[0], boundary[1], color = "#a6dba0", alpha = 0.45, zorder = 1 )
    ax.plot ( [boundary[1], boundary[1]], [y_min, y_max], color = "black", linestyle = "--", zorder = 2 )
    ax.axvspan ( boundary[1], x_max, color = "#92c5de", alpha = 0.45, zorder = 1 )

  ## scatter plot
  if len(labels) == 2:
    tn_idx = np.nonzero ( y_true == False )   # true negative
    tp_idx = np.nonzero ( y_true == True )    # true positive
    ax.scatter ( y_scores[:,1][tn_idx], X_feat[tn_idx], 
                  color = "#ca0020", marker = "o", label = labels[0], zorder = 3 )
    ax.scatter ( y_scores[:,1][tp_idx], X_feat[tp_idx], 
                  color = "#0571b0", marker = "^", label = labels[1], zorder = 4 )
  elif len(labels) == 3:
    cls1_idx = np.nonzero ( y_true == np.unique(y_true)[0] )
    cls2_idx = np.nonzero ( y_true == np.unique(y_true)[1] )
    cls3_idx = np.nonzero ( y_true == np.unique(y_true)[2] )
    ax.scatter ( y_scores[:,1][cls1_idx], X_feat[cls1_idx], 
                 color = "#ca0020", marker = "o", label = labels[0], zorder = 3 )
    ax.scatter ( y_scores[:,1][cls3_idx], X_feat[cls3_idx], 
                 color = "#0571b0", marker = "^", label = labels[2], zorder = 4 )
    ax.scatter ( y_scores[:,1][cls2_idx], X_feat[cls2_idx], 
                 color = "#008837", marker = "s", label = labels[1], zorder = 5 )

  ax.legend (title = "True label", loc = "upper left", fontsize = 10)
  ax.axis ([x_min, x_max, y_min, y_max])
  
  ## figure name default
  if fig_name is None:
    timestamp = str (datetime.now()) . split (".") [0]
    timestamp = timestamp . replace (" ","_")
    fig_name = "clf_region_"
    for time, unit in zip ( timestamp.split(":"), ["h","m","s"] ):
      fig_name += time + unit   # YYYY-MM-DD_HHhMMmSSs
  filename = f"docs/img/{fig_name}.png"
  
  plt.tight_layout()
  if save_figure: 
    plt.savefig ( filename, format = "png", dpi = RESOLUTION )
    print (f"Figure correctly exported to {filename}")

  plt.show()
