import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser

RESOLUTION = 100

#   +-------------------+
#   |   Options setup   |
#   +-------------------+

parser = ArgumentParser (description = "models comparison")
parser . add_argument ("-t", "--threshold", default = "rec80")
args = parser.parse_args()

#   +---------------+
#   |   File list   |
#   +---------------+

score_dir = "scores"

sort_keys = [ "log-reg" , "lin-svm" , "gaus-proc" , "rnd-frs" , "grad-bdt" ]
file_list = [ f"bin-clf_{k}_{args.threshold}.npz" for k in sort_keys ]

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

#   +------------------+
#   |   Figure setup   |
#   +------------------+

fig, ax = plt.subplots (figsize = (8,6), dpi = RESOLUTION)
ax.set_title ("ROC curves of trained models", fontsize = 14)
ax.set_xlabel ("True Negative Rate", fontsize = 12)
ax.set_ylabel ("True Positive Rate", fontsize = 12)

colors = [ "#ff7f00" , "#984ea3" , "#4daf4a" ,  "#377eb8" , "#e41a1c" ]

#   +---------------------+
#   |   ROC curve plots   |
#   +---------------------+

ax.plot ([0,1], [1,0], color = "black", linestyle = "--")

for z, (npz_file, color) in enumerate ( zip(file_list, colors) ):
  model_name  = extract_name (npz_file)
  score_values  = np.load (f"{score_dir}/{npz_file}") ["roc_vars"]
  tnr, tpr, auc_mean, auc_std = score_values[:,0], score_values[:,1], score_values[0,2], score_values[0,3]

  ax.plot (tnr, tpr, color = color, label = f"[AUC : ${auc_mean:.2f} \pm {auc_std:.2f}$] {model_name}", zorder = z)

ax.legend (loc = "lower left", fontsize = 10)

#   +---------------------+
#   |   Save the figure   |
#   +---------------------+

filename = f"docs/img/bin-clf/roc_curves_{args.threshold}.png"

plt.tight_layout()
plt.savefig ( filename, format = "png", dpi = RESOLUTION )
print (f"Figure correctly exported to {filename}")

plt.show()
