import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser

RESOLUTION = 300
LABELS = ["PMBCL", "GZL"]

#   +-------------------+
#   |   Options setup   |
#   +-------------------+

parser = ArgumentParser (description = "models comparison")
parser . add_argument ("-t", "--threshold", default = "rec90")
args = parser.parse_args()

#   +---------------+
#   |   File list   |
#   +---------------+

score_dir = "scores"

sort_keys = [ "log-reg", "lin-svm", "gaus-proc", "rnd-frs", "grad-bdt", "suv-max" ]
file_list = [ f"{k}_{args.threshold}.npz" for k in sort_keys ]

#   +---------------------------+
#   |   Model name extraction   |
#   +---------------------------+

def extract_name (filename) -> str:
  model_name = filename.split("_")[0]
  if   model_name == "log-reg"   : return "Logistic Regression"
  elif model_name == "lin-svm"   : return "Linear SVM classifier"
  elif model_name == "gaus-proc" : return "Gaussian Process classifier"
  elif model_name == "rnd-frs"   : return "Random Forest classifier"
  elif model_name == "grad-bdt"  : return "Gradient BDT classifier"
  elif model_name == "suv-max"   : return "SUV$_{max}$-based classifier"

#   +------------------+
#   |   Figure setup   |
#   +------------------+

colors = [ "#0571b0", "#ca0020", "#7b3294", "#4dac26", "#e9a3c9", "#fdb863" ]

fig, ax = plt.subplots (nrows = 1, ncols = 3, figsize = (24,6), dpi = RESOLUTION)

for i in range(3):

  if (i == 0):
    print ( "\n\t\t\t+-------------------------------------------+"   )
    print ( "  \t\t\t|                                           |"   )
    print ( "  \t\t\t|     Performance of binary classifiers     |"   )
    print ( "  \t\t\t|                                           |"   )
    print ( "  \t\t\t+-------------------------------------------+\n" )
    ax[i].set_title (f"ROC curves for binary classification", fontsize = 14)
  else:
    print ( "\n\t\t\t+-----------------------------------------------+"   )
    print ( "  \t\t\t|                                               |"   )
    print ( "  \t\t\t|     Performance of multiclass classifiers     |"   )
    print ( "  \t\t\t|                                               |"   )
    print ( "  \t\t\t+-----------------------------------------------+\n" )
    ax[i].set_title (f"One-vs-all ROC curves for {LABELS[i-1]} classification", fontsize = 14)

  ax[i].set_xlabel ("Specificity", fontsize = 12)
  ax[i].set_ylabel ("Sensitivity", fontsize = 12)

  #   +---------------------+
  #   |   ROC curve plots   |
  #   +---------------------+

  ax[i].plot ([0,1], [1,0], color = "black", linestyle = "--")

  for z, (file_name, color) in enumerate ( zip(file_list, colors) ):

    model_name  = extract_name (file_name)
    print ( f"  {model_name}\n  " + "=" * 40 )

    if (i == 0):
      npz_loaded = np.load (f"{score_dir}/bin-clf/{file_name}")
      roc_curves = npz_loaded["roc"]
      auc_scores = npz_loaded["auc"]
      print ( f"    AUC : {auc_scores[2]:.2f} +/- {auc_scores[3]:.2f}" )
      print ( f"    AUC (10th percentile) : {auc_scores[0]:.2f}" )
      print ( f"    AUC (32nd percentile) : {auc_scores[1]:.2f}\n" )
      
    else:
      npz_loaded = np.load (f"{score_dir}/multi-clf/{file_name}")
      roc_curves = npz_loaded[f"roc_{LABELS[i-1].lower()}"]
      auc_scores = npz_loaded[f"auc_{LABELS[i-1].lower()}"]
      print ( f"    {LABELS[i-1]} AUC : {auc_scores[2]:.2f} +/- {auc_scores[3]:.2f}" )
      print ( f"    {LABELS[i-1]} AUC (10th percentile) : {auc_scores[0]:.2f}" )
      print ( f"    {LABELS[i-1]} AUC (32nd percentile) : {auc_scores[1]:.2f}\n" )

    ax[i].plot (roc_curves[:,0], roc_curves[:,1], color = color, label = f"[AUC : ${auc_scores[2]:.2f} \pm {auc_scores[3]:.2f}$] {model_name}", zorder = z)

  ax[i].legend (loc = "lower left", fontsize = 10)

#   +---------------------+
#   |   Save the figure   |
#   +---------------------+

filename = f"docs/img/roc_curves_{args.threshold}.png"
plt.savefig ( filename, format = "png", dpi = RESOLUTION )
print (f"Figure correctly exported to {filename}")

plt.show()
