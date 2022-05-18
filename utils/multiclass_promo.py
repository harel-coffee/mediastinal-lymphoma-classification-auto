import numpy as np

from sklearn.metrics import confusion_matrix


def multiclass_promo ( y_true   : np.ndarray ,
                       y_scores : np.ndarray ,
                       boundaries : tuple = (0.5, 0.5) ) -> np.ndarray:
  class1_idx = np.nonzero ( y_scores[:,1] <  boundaries[0] )   # predicted class 1
  class3_idx = np.nonzero ( y_scores[:,1] >= boundaries[1] )   # predicted class 3
  y_pred = 2 * np.ones_like ( y_scores[:,1] )
  y_pred[class1_idx] = 1.
  y_pred[class3_idx] = 3.
  return confusion_matrix ( y_true, y_pred )
  