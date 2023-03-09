#  Run with:
#    python multiclass_pr_roc_curves.py -t rec90


import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser

RESOLUTION = 600
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

score_dir = "./scores"

sort_keys = [ "log-reg", "lin-svm", "gaus-proc", "rnd-frs", "grad-bdt", "suv-max" ]
file_list = [ f"multi-clf_{k}_{args.threshold}.npz" for k in sort_keys ]

#   +---------------------------+
#   |   Model name extraction   |
#   +---------------------------+

def extract_name (filename) -> str:
  model_name = filename.split("_")[1]
  if   model_name == "log-reg"   : return "Logistic Regression"
  elif model_name == "lin-svm"   : return "Linear SVM classifier"
  elif model_name == "gaus-proc" : return "Gaussian Process classifier"
  elif model_name == "rnd-frs"   : return "Random Forest classifier"
  elif model_name == "grad-bdt"  : return "Gradient BDT classifier"
  elif model_name == "suv-max"   : return "SUV$_{max}$-based classifier"

#   +------------------+
#   |   Figure setup   |
#   +------------------+

print ( "\n\t\t\t+-----------------------------------------------+"   )
print ( "  \t\t\t|                                               |"   )
print ( "  \t\t\t|     Performance of multiclass classifiers     |"   )
print ( "  \t\t\t|                                               |"   )
print ( "  \t\t\t+-----------------------------------------------+\n" )

colors = [ "#0571b0", "#ca0020", "#7b3294", "#4dac26", "#e9a3c9", "#fdb863" ]

for label in LABELS:

  print ("\t\t\t\t\t<<< {label} >>>")

  fig, ax = plt.subplots (nrows = 1, ncols = 2, figsize = (16,6), dpi = RESOLUTION)

  for i in range(2):
    if (i==0):
      ax[i].set_title  (f"One-vs-all ROC curves for {label} classification", fontsize = 14)
      ax[i].set_xlabel ("Specificity", fontsize = 12)
      ax[i].set_ylabel ("Sensitivity", fontsize = 12)
    else:
      ax[i].set_title  (f"One-vs-all Precision-Recall curves for {label} classification", fontsize = 14)
      ax[i].set_xlabel ("Recall", fontsize = 12)
      ax[i].set_ylabel ("Precision", fontsize = 12)

    #   +------------------------+
    #   |   PR/ROC curve plots   |
    #   +------------------------+

    for z, (file_name, color) in enumerate ( zip(file_list, colors) ):

      npz_loaded = np.load (f"{score_dir}/{file_name}")
      baseline = npz_loaded[f"baseline_{label.lower()}"]
      roc_curves = npz_loaded[f"roc_{label.lower()}"]
      auc_scores = npz_loaded[f"auc_{label.lower()}"]
      pr_curves  = npz_loaded[f"pr_{label.lower()}"]
      ap_scores  = npz_loaded[f"ap_{label.lower()}"]

      model_name  = extract_name (file_name)

      if (i == 0):
        print ( f"  {model_name}\n  " + "=" * 40 )
        print ( f"    {label} AUC : {auc_scores[2]:.2f} +/- {auc_scores[3]:.2f}" )
        print ( f"    {label} AUC (10th percentile) : {auc_scores[0]:.2f}" )
        print ( f"    {label} AUC (32nd percentile) : {auc_scores[1]:.2f}" )
        print ( "  " + "-" * 32 )
        print ( f"    {label} AP : {ap_scores[2]:.2f} +/- {ap_scores[3]:.2f}" )
        print ( f"    {label} AP (10th percentile) : {ap_scores[0]:.2f}" )
        print ( f"    {label} AP (32nd percentile) : {ap_scores[1]:.2f}\n" )

        ax[i].plot ([0,1], [1,0], color = "black", linestyle = "--")
        ax[i].plot (roc_curves[:,0], roc_curves[:,1], color = color, 
                    label = f"[AUC : ${auc_scores[2]:.2f} \pm {auc_scores[3]:.2f}$] {model_name}",
                    zorder = z)
      else:
        ax[i].plot ([0,1], [baseline,baseline], color = "black", linestyle = "--")
        ax[i].plot (pr_curves[:,0], pr_curves[:,1], color = color, 
                    label = f"[AP : ${ap_scores[2]:.2f} \pm {ap_scores[3]:.2f}$] {model_name}",
                    zorder = z)

    ax[i].legend (fontsize = 10)

  #   +---------------------+
  #   |   Save the figure   |
  #   +---------------------+

  filename = f"./img/multi-clf/roc_curves_{label.lower()}_{args.threshold}.png"
  plt.savefig ( filename, format = "png", dpi = RESOLUTION )
  print (f"Figure correctly exported to {filename}")

  plt.show()
  plt.close()
