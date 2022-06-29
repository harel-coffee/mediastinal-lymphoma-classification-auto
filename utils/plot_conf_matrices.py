import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime

RESOLUTION = 300


def plot_conf_matrices ( conf_matrix : np.ndarray    , 
                         labels      : list = None   ,
                         show_matrix : str  = "both" ,
                         save_figure : bool = False  ,
                         fig_name    : str  = None   ) -> None:
  precision = np.zeros ( len(conf_matrix) )
  recall    = np.zeros ( len(conf_matrix) )

  for i in range ( len(conf_matrix) ):
    precision[i] = conf_matrix[i,i] / np.sum ( conf_matrix[:,i] )
    recall[i]    = conf_matrix[i,i] / np.sum ( conf_matrix[i,:] )

  tmp = ""
  timestamp = str (datetime.now()) . split (".") [0]
  timestamp = timestamp . replace (" ","_")
  for time, unit in zip ( timestamp.split(":"), ["h","m","s"] ):
    tmp += time + unit   # YYYY-MM-DD_HHhMMmSSs
  
  if show_matrix in ["std", "both"]:
    plt.figure ( figsize = (5,5), dpi = RESOLUTION )    
    plt.title ( "Confusion matrix", fontsize = 14 )
    df_conf_mtx = pd.DataFrame (conf_matrix, index = labels, columns = labels )
    sns.heatmap ( df_conf_mtx, annot = True, annot_kws = { "size" : 14 }, cmap = "Blues" )
    plt.xlabel ( "Predicted labels", fontsize = 12 )
    plt.ylabel ( "True labels", fontsize = 12)
    filename = f"{fig_name}_conf_matrix" if fig_name else f"conf_matrix_{tmp}"
    filename = f"docs/img/{filename}.png"
    if save_figure:
      plt.savefig ( filename, format = "png", dpi = RESOLUTION )
      print (f"Figure correctly exported to {filename}")
    plt.show()

  if show_matrix in ["norm", "both"]:
    plt.figure ( figsize = (5,5), dpi = RESOLUTION )
    plt.title ( "Normalized confusion matrix", fontsize = 14 )
    norm_conf_matrix = conf_matrix / np.sum ( conf_matrix, axis = 1 ) [:,None]
    df_norm_conf_mtx = pd.DataFrame (norm_conf_matrix, index = labels, columns = labels )
    sns.heatmap ( df_norm_conf_mtx, annot = True, annot_kws = { "size" : 14 }, cmap = "Blues" )
    plt.xlabel ( "Predicted labels", fontsize = 12 )
    plt.ylabel ( "True labels", fontsize = 12)
    filename = f"{fig_name}_norm_conf_matrix" if fig_name else f"norm_conf_matrix_{tmp}"
    filename = f"docs/img/{filename}.png"
    if save_figure:
      plt.savefig ( filename, format = "png", dpi = RESOLUTION )
      print (f"Figure correctly exported to {filename}")
    plt.show()
  