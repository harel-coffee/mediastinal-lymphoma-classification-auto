import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

RESOLUTION = 100


def plot_prf_histos ( tpr_scores  : np.ndarray   ,
                      tnr_scores  : np.ndarray   ,
                      bins        : int  = 100   ,
                      title       : str  = None  ,
                      save_figure : bool = False ,
                      fig_name    : str  = None  ) -> None:
  ## figure setup
  fig, ax = plt.subplots (figsize = (8,5), dpi = RESOLUTION)
  ax.set_title  (title, fontsize = 14)
  ax.set_xlabel ("Score", fontsize = 12)
  ax.set_ylabel ("Entries", fontsize = 12)

  ax.hist ( tpr_scores, bins = bins, range = [0,1], histtype = "stepfilled", edgecolor = "darkgreen",
             lw = 1.5, color = "#1a9850", alpha = 0.7, label = "True Positive Rate (TPR)", zorder = 1 )
  ax.hist ( tnr_scores, bins = bins, range = [0,1], histtype = "stepfilled", edgecolor = "darkred",
             lw = 1.5, color = "#d73027", alpha = 0.7, label = "True Negative Rate (TNR)", zorder = 0 )

  ax.annotate ( f"TPR : ${np.mean(tpr_scores):.2f} \pm {np.std(tpr_scores):.2f}$", 
                 ha = "left", va = "top", xy = (0.10, 0.80), xycoords = "axes fraction", fontsize = 12 )
  ax.annotate ( f"TNR : ${np.mean(tnr_scores):.2f} \pm {np.std(tnr_scores):.2f}$", 
                 ha = "left", va = "top", xy = (0.10, 0.74), xycoords = "axes fraction", fontsize = 12 )

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
