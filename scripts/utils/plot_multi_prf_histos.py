import os
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

RESOLUTION = 600


def plot_multi_prf_histos ( tpr_scores  : tuple ,
                            tnr_scores  : tuple ,
                            ppv_scores  : tuple ,
                            bins        : int   = 100  ,
                            title       : str   = None ,
                            cls_labels  : tuple = None ,
                            save_figure : bool = False ,
                            fig_name    : str  = None  ) -> None:
  ## figure setup
  fig, ax = plt.subplots (figsize = (8,6), dpi = RESOLUTION)
  ax.set_title  (title, fontsize = 14)
  ax.set_xlabel ("Score", fontsize = 12)
  ax.set_ylabel ("Entries", fontsize = 12)

  tpr = tpr_scores[0][~np.isnan(tpr_scores[0])]
  tpr_10pctl, tpr_32pctl, tpr_mean = _get_scores_to_plot (tpr)

  tnr = tnr_scores[0][~np.isnan(tnr_scores[0])]
  tnr_10pctl, tnr_32pctl, tnr_mean = _get_scores_to_plot (tnr)

  ppv = ppv_scores[0][~np.isnan(ppv_scores[0])]
  ppv_10pctl, ppv_32pctl, ppv_mean = _get_scores_to_plot (ppv)

  ax.hist ( tpr, bins = bins, range = [0,1], histtype = "stepfilled", 
            edgecolor = "darkblue", lw = 1.5, color = "#377eb8", alpha = 0.7, 
            label = f"One-vs-all TPR for {cls_labels[0]} : {tpr_mean:.2f} [{tpr_10pctl:.2f}, {tpr_32pctl:.2f}]", 
            zorder = 3 )
  ax.hist ( tnr, bins = bins, range = [0,1], histtype = "stepfilled", 
            edgecolor = "darkred", lw = 1.5, color = "#e41a1c", alpha = 0.7, 
            label = f"One-vs-all TNR for {cls_labels[0]} : {tnr_mean:.2f} [{tnr_10pctl:.2f}, {tnr_32pctl:.2f}]", 
            zorder = 1 )
  ax.hist ( ppv, bins = bins, range = [0,1], histtype = "stepfilled", 
            edgecolor = "darkgreen", lw = 1.5, color = "#1a9850", alpha = 0.7, 
            label = f"One-vs-all PPV for {cls_labels[0]} : {ppv_mean:.2f} [{ppv_10pctl:.2f}, {ppv_32pctl:.2f}]", 
            zorder = 1 )

  tpr = tpr_scores[1][~np.isnan(tpr_scores[1])]
  tpr_10pctl, tpr_32pctl, tpr_mean = _get_scores_to_plot (tpr)

  tnr = tnr_scores[1][~np.isnan(tnr_scores[1])]
  tnr_10pctl, tnr_32pctl, tnr_mean = _get_scores_to_plot (tnr)

  ppv = ppv_scores[1][~np.isnan(ppv_scores[1])]
  ppv_10pctl, ppv_32pctl, ppv_mean = _get_scores_to_plot (ppv)

  ax.hist ( tpr, bins = bins, range = [0,1], histtype = "stepfilled", 
            edgecolor = "purple", lw = 1.5, color = "#984ea3", alpha = 0.7, 
            label = f"One-vs-all TPR for {cls_labels[1]} : {tpr_mean:.2f} [{tpr_10pctl:.2f}, {tpr_32pctl:.2f}]", 
            zorder = 2 )
  ax.hist ( tnr, bins = bins, range = [0,1], histtype = "stepfilled", 
            edgecolor = "sienna", lw = 1.5, color = "#ff7f00", alpha = 0.7, 
            label = f"One-vs-all TNR for {cls_labels[1]} : {tnr_mean:.2f} [{tnr_10pctl:.2f}, {tnr_32pctl:.2f}]", 
            zorder = 0 )
  ax.hist ( ppv, bins = bins, range = [0,1], histtype = "stepfilled", 
            edgecolor = "black", lw = 1.5, color = "#bababa", alpha = 0.7, 
            label = f"One-vs-all PPV for {cls_labels[1]} : {ppv_mean:.2f} [{ppv_10pctl:.2f}, {ppv_32pctl:.2f}]", 
            zorder = 1 )

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
