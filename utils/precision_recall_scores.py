import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime        import datetime
from sklearn.metrics import confusion_matrix

from .save_figure import save_figure


def precision_recall_scores ( y_true : np.ndarray , 
                              y_pred : np.ndarray ,
                              labels : list = None , 
                              show_conf_matrix : bool = False ,
                              fig_name : str = None ,
                              verbose : bool = False ) -> tuple:
  if labels is not None:
    if len(labels) != len(np.unique(y_true)):
      raise ValueError ( "Labels list length doesn't match with the true number of classes." )
  else:
    labels = np.unique ( y_true )

  conf_matrix = confusion_matrix ( y_true, y_pred )
  precision = np.zeros ( len(conf_matrix) )
  recall    = np.zeros ( len(conf_matrix) )

  for i in range ( len(conf_matrix) ):
    precision[i] = conf_matrix[i,i] / np.sum ( conf_matrix[:,i] )
    recall[i]    = conf_matrix[i,i] / np.sum ( conf_matrix[i,:] )
    if verbose:
      print ( "+---->  Label {:3}  <----+" . format (labels[i]) )
      print ( "|   Precision : {:.1f}%   |" . format (100 * precision[i]) )
      print ( "|   Recall    : {:.1f}%   |" . format (100 * recall[i])    )
  if verbose: print ( "+-----------------------+" )

  if show_conf_matrix:
    if fig_name is None:
      timestamp = str (datetime.now()) . split (".") [0]
      timestamp = timestamp . replace (" ","_")
      fig_name = "precision_recall_scores_"
      for time, unit in zip ( timestamp.split(":"), ["h","m","s"] ):
        fig_name += time + unit   # YYYY-MM-DD_HHhMMmSSs

    plt.figure ( figsize = (5,5), dpi = 100 )    
    plt.title ( "Confusion matrix", fontsize = 14 )
    df_conf_mtx = pd.DataFrame (conf_matrix, index = labels, columns = labels )
    sns.heatmap ( df_conf_mtx, annot = True, annot_kws = { "size" : 14 }, cmap = "Blues" )
    plt.xlabel ( "Predicted labels", fontsize = 12 )
    plt.ylabel ( "True labels", fontsize = 12)
    save_figure ( f"{fig_name}_conf_matrix" )
    plt.show()

    plt.figure ( figsize = (5,5), dpi = 100 )
    plt.title ( "Normalized confusion matrix", fontsize = 14 )
    norm_conf_matrix = conf_matrix / np.sum ( conf_matrix, axis = 1 ) [:,None]
    df_norm_conf_mtx = pd.DataFrame (norm_conf_matrix, index = labels, columns = labels )
    sns.heatmap ( df_norm_conf_mtx, annot = True, annot_kws = { "size" : 14 }, cmap = "Blues" )
    plt.xlabel ( "Predicted labels", fontsize = 12 )
    plt.ylabel ( "True labels", fontsize = 12)
    save_figure ( f"{fig_name}_norm_conf_matrix" )
    plt.show()

  return precision, recall
  