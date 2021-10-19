import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator


class DummyClassifier ( BaseEstimator ):
  def __init__ ( self ):
    super().__init__()
    self._labels = None
    self._lbl_ratios = None

  def fit ( self, X, y ):
    if not isinstance ( X, (tuple, list, np.ndarray, pd.Series, pd.DataFrame) ):
      raise ValueError ( "X should be an array." )
    else:
      X = np.array ( X )

    if not isinstance ( y, (tuple, list, np.ndarray, pd.Series) ):
      raise ValueError ( "y should be an array." )
    else:
      y = np.array ( y )

    if len ( y.shape ) != 1:
      raise ValueError ( "y should be a 1-D array." )

    if len(X) != len(y):
      raise ValueError ( "X and y should have the same length." )

    self._labels = np.unique ( y )
    self._lbl_ratios = [ len ( np.nonzero ( y == lbl ) [0] ) / len ( y ) for lbl in self._labels ]

  def predict ( self, X ):
    edges = np.cumsum ( self._lbl_ratios )
    rnd_nums = np.random.rand ( len(X) )
    rnd_matrix = np.diag ( rnd_nums ) @ np.ones ( shape = (len(X), len(edges)) )

    edges_idx = np.array ( np.nonzero ( rnd_matrix < edges ) )
    left_edges_idx = np.unique ( edges_idx[0], return_index = True ) [1]
    lbl_idx = edges_idx [ 1, left_edges_idx ]

    predictions = [ self._labels[i] for i in lbl_idx ]

    return np.array ( predictions )
