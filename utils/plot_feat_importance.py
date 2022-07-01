import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime

RESOLUTION = 300


def plot_feat_importance ( feat_ranks  : np.ndarray   ,
                           feat_names  : list = None  ,
                           save_figure : bool = False ,
                           fig_name    : str  = None  ) -> None:
  ## default values
  num_feats = feat_ranks.shape[1]
  if feat_names is None: feat_names = [ f"feature_{i+1}" for i in range(num_feats) ]

  ## ranking counts
  rank_counts = list()
  for idx in range (num_feats):
    mask = ( feat_ranks == idx )
    rank_counts . append ( np.sum (mask, axis = 0) )
  rank_counts = np.array ( rank_counts ) / feat_ranks.shape[0]   # normalized

  ## re-ordering of features
  avg_rank = np.sum ( rank_counts * np.arange(1,num_feats+1), axis = 1 ) / np.sum ( rank_counts, axis = 1 )
  feat_ord = np.argsort ( avg_rank )
  feat_names = np.array ( feat_names )
  df = pd.DataFrame ( rank_counts[feat_ord,:], index = feat_names[feat_ord], columns = [f"#{i+1}" for i in range(num_feats)] )

  ## figure setup
  fig, ax = plt.subplots (figsize = (8,5), dpi = RESOLUTION)
  sns.heatmap ( df, annot = True, annot_kws = { "size" : 12 }, cmap = "mako" )
  ax.set_xlabel ("Feature ranking", fontsize = 12)

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
