import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from argparse import ArgumentParser

RESOLUTION = 600


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
  img_dir = "./img"
  filename = f"{img_dir}/{fig_name}.png"

  plt.tight_layout()
  if save_figure:
    if not os.path.exists(img_dir):
      os.makedirs(img_dir)
    plt.savefig ( filename, format = "png", dpi = RESOLUTION )
    print (f"Figure correctly exported to {filename}")

  plt.show()


if __name__ == "__main__":
  parser = ArgumentParser ( description = "feature importance" )
  parser . add_argument ( "-f" , "--filename" , required = True )
  args = parser . parse_args()

  npz_file = np.load (args.filename)
  feat_ranks = npz_file["ranks"]
  feat_names = npz_file["names"]

  plot_feat_importance ( feat_ranks  = feat_ranks ,
                         feat_names  = feat_names ,
                         save_figure = True )
