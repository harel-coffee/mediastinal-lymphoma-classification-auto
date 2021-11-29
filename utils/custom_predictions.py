import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve


def custom_predictions ( y_true : np.ndarray ,
                         y_scores : np.ndarray ,
                         recall_score : float = None ,
                         precision_score : float = None ,
                         show_curves : bool = False ) -> tuple:
  if len(y_scores.shape) == 2:
    y_scores = y_scores[:,1]

  precision, recall, threshold = precision_recall_curve (y_true, y_scores)
  if show_curves:
    plt.figure (figsize = (8,5), dpi = 100)
    plt.xlabel ("Threshold", fontsize = 12)
    plt.plot (threshold, precision[:-1], color = "coral", linestyle = "--", label = "Precision")
    plt.plot (threshold, recall[:-1], color = "dodgerblue", linestyle = "-", label = "Recall")
    plt.legend (loc = "lower left", fontsize = 12)
    plt.show()

  if (recall is not None ) and (precision_score is None):
    if (recall_score < 0) or (recall_score > 1):
      raise ValueError ("The recall score should be less than 1.")
    custom_threshold = threshold [np.argmin (recall >= recall_score) - 1]
    return (y_scores >= custom_threshold), custom_threshold

  elif (precision is not None) and (recall_score is None):
    if (precision_score < 0) or (precision_score > 1):
      raise ValueError ("The precision score should be less than 1.")
    custom_threshold = threshold [np.argmax (precision >= precision_score)]
    return (y_scores >= custom_threshold), custom_threshold

  elif (precision_score is None) and (recall_score is None):
    custom_threshold = threshold [np.argmin (recall > precision)]
    return (y_scores >= custom_threshold), custom_threshold

  else:
    raise ValueError ("Only one target score should be passed: recall or precision.")
