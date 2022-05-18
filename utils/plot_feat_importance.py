import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

RESOLUTION = 100


def plot_feat_importance ( feat_scores : np.ndarray   ,
                           feat_errors : np.ndarray   ,
                           feat_names  : list = None  ,
                           title       : str  = None  ,
                           save_figure : bool = False ,
                           fig_name    : str  = None  ) -> None:
  y_pos = np.arange(len(feat_scores))

  ## default values
  if feat_names is None: feat_names = [ f"feature_{i}" for i in range(len(feat_scores)) ]
  if title is None: title = "Feature importances"

  ## figure setup
  fig, ax = plt.subplots (figsize = (8,5), dpi = RESOLUTION)
  ax.set_title  (title, fontsize = 14)
  ax.set_xlabel ("Score", fontsize = 12)
  ax.barh (y_pos, feat_scores, xerr = feat_errors, capsize = 4, align = "center")
  ax.set_yticks (y_pos)
  ax.set_yticklabels (labels = feat_names, fontsize = 12)
  ax.invert_yaxis()

  ## figure name default
  if fig_name is None:
    timestamp = str (datetime.now()) . split (".") [0]
    timestamp = timestamp . replace (" ","_")
    fig_name = "feat_importance_"
    for time, unit in zip ( timestamp.split(":"), ["h","m","s"] ):
      fig_name += time + unit   # YYYY-MM-DD_HHhMMmSSs
  filename = f"docs/img/{fig_name}.png"

  plt.tight_layout()
  if save_figure: plt.savefig ( filename, format = "png", dpi = RESOLUTION )
  print (f"Figure correctly exported to {filename}")

  plt.show()
