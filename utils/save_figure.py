import matplotlib.pyplot as plt


def save_figure ( fig_name, resolution = 300, verbose = False ):
  path = "docs/img"
  filename = f"{path}/{fig_name}.png"
  plt.tight_layout()
  plt.savefig ( filename, format = "png", dpi = resolution )
  if verbose: print ( f"Figure correctly exported to {filename}." )
