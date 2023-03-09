import os
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

RESOLUTION = 600


def plot_bin_prf_histos ( tpr_scores  : np.ndarray   ,
                          tnr_scores  : np.ndarray   ,
                          bins        : int  = 100   ,
                          title       : str  = None  ,
                          save_figure : bool = False ,
                          fig_name    : str  = None  ) -> None:
  ## figure setup
  fig, ax = plt.subplots (figsize = (8,6), dpi = RESOLUTION)
  ax.set_title  (title, fontsize = 14)
  ax.set_xlabel ("Score", fontsize = 12)
  ax.set_ylabel ("Entries", fontsize = 12)

  tpr = tpr_scores[~np.isnan(tpr_scores)]
  tpr_10pctl, tpr_32pctl, tpr_mean = _get_scores_to_plot (tpr)

  ax.hist ( tpr, bins = bins, range = [0,1], histtype = "stepfilled", 
            edgecolor = "darkgreen", lw = 1.5, color = "#1a9850", alpha = 0.7, 
            label = f"TPR : {tpr_mean:.2f} [{tpr_10pctl:.2f}, {tpr_32pctl:.2f}]", 
            zorder = 1 )

  tnr = tnr_scores[~np.isnan(tnr_scores)]
  tnr_10pctl, tnr_32pctl, tnr_mean = _get_scores_to_plot (tnr)
  
  ax.hist ( tnr, bins = bins, range = [0,1], histtype = "stepfilled", 
            edgecolor = "darkred", lw = 1.5, color = "#d73027", alpha = 0.7, 
            label = f"TNR : {tnr_mean:.2f} [{tnr_10pctl:.2f}, {tnr_32pctl:.2f}]", 
            zorder = 0 )

  ax.legend (loc = "upper left", fontsize = 12)

  ## figure name default
  if fig_name is None:
    timestamp = str (datetime.now()) . split (".") [0]
    timestamp = timestamp . replace (" ","_")
    fig_name = "prf_histos_"
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


def _get_scores_to_plot (score):
  pctl_10 = np.percentile ( score, 10, axis = 0 )
  pctl_32 = np.percentile ( score, 32, axis = 0 )
  mean = np.mean ( score, axis = 0 )
  return pctl_10, pctl_32, mean
