import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

RESOLUTION = 100


def plot_multi_prf_histos ( rec_scores  : tuple ,
                            prec_scores : tuple ,
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

  ax.hist ( rec_scores[0], bins = bins, range = [0,1], histtype = "stepfilled", edgecolor = "darkblue", lw = 1.5, color = "#377eb8", alpha = 0.7, 
            label = f"[{cls_labels[0]} class] Recall : ${np.mean(rec_scores[0]):.2f} \pm {np.std(rec_scores[0]):.2f}$", zorder = 3 )
  ax.hist ( prec_scores[0], bins = bins, range = [0,1], histtype = "stepfilled", edgecolor = "darkred", lw = 1.5, color = "#e41a1c", alpha = 0.7, 
            label = f"[{cls_labels[0]} class] Precision : ${np.mean(prec_scores[0]):.2f} \pm {np.std(prec_scores[0]):.2f}$", zorder = 1 )

  ax.hist ( rec_scores[1], bins = bins, range = [0,1], histtype = "stepfilled", edgecolor = "purple", lw = 1.5, color = "#984ea3", alpha = 0.7, 
            label = f"[{cls_labels[1]} class] Recall : ${np.mean(rec_scores[1]):.2f} \pm {np.std(rec_scores[1]):.2f}$", zorder = 2 )
  ax.hist ( prec_scores[1], bins = bins, range = [0,1], histtype = "stepfilled", edgecolor = "sienna", lw = 1.5, color = "#ff7f00", alpha = 0.7, 
            label = f"[{cls_labels[1]} class] Precision : ${np.mean(rec_scores[1]):.2f} \pm {np.std(rec_scores[1]):.2f}$", zorder = 0 )

  ax.legend (loc = "upper left", fontsize = 12)

  ## figure name default
  if fig_name is None:
    timestamp = str (datetime.now()) . split (".") [0]
    timestamp = timestamp . replace (" ","_")
    fig_name = "prf_histos_"
    for time, unit in zip ( timestamp.split(":"), ["h","m","s"] ):
      fig_name += time + unit   # YYYY-MM-DD_HHhMMmSSs
  filename = f"docs/img/{fig_name}.png"

  plt.tight_layout()
  if save_figure: plt.savefig ( filename, format = "png", dpi = RESOLUTION )
  print (f"Figure correctly exported to {filename}")

  plt.show()
