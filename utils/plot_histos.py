import pandas as pd
import matplotlib.pyplot as plt


def plot_histos ( df : pd.DataFrame ,
                  vars : list ,
                  n_rows : int ,
                  n_cols : int ,
                  plot_style : str = "overlapped" , 
                  figsize : tuple = (10,6) , 
                  legend_on_left : list = [] ) -> None:
  if len(vars) != ( n_rows * n_cols ):
    raise ValueError (
      "Histograms for the labels passed cannot be arrange as a grid of rows = %d and cols = %d." % (n_rows, n_cols)
                     ) 
  
  fig, ax = plt.subplots (n_rows, n_cols, figsize = figsize)
  plt.subplots_adjust (wspace = 0.2, hspace = 0.2)

  labels = ["HL", "GZ", "PML"]
  colors = ["dodgerblue", "deeppink", "forestgreen"]

  ### Histos in 2-D array
  if (n_rows != 1) and (n_cols != 1):

    n = 0
    for i in range(n_rows):
      for j in range(n_cols):

        x_1 = df[df["lymphoma_type"]==1][vars[n]]   # label "HL"
        x_2 = df[df["lymphoma_type"]==2][vars[n]]   # label "GZ"
        x_3 = df[df["lymphoma_type"]==3][vars[n]]   # label "PML"

        ax[i,j] . set_title ( vars[n][:40], size = 10 )

        if plot_style == "stacked":
          ax[i,j] . hist ( [x_1, x_2, x_3], bins = 20, stacked = True, color = colors, label = labels )

        elif plot_style == "overlapped":
          r_min = min ( x_1.min(), x_2.min(), x_3.min() )
          r_max = max ( x_1.max(), x_2.max(), x_3.max() )
          ax[i,j] . hist ( x_1, range = (r_min, r_max), bins = 20, alpha = 0.7, color = "dodgerblue" , label = "HL"  )
          ax[i,j] . hist ( x_2, range = (r_min, r_max), bins = 20, alpha = 0.7, color = "deeppink"   , label = "GZ"  )
          ax[i,j] . hist ( x_3, range = (r_min, r_max), bins = 20, alpha = 0.7, color = "forestgreen", label = "PML" )
          ax[i,j] . grid ( True, alpha = 0.3 )

        else:
          pass   # empty plots

        if plot_style in ["stacked", "overlapped"]:
          if n in legend_on_left:
            ax[i,j] . legend ( loc = "upper left" , fontsize = 8 )  
          else:
            ax[i,j] . legend ( loc = "upper right", fontsize = 8 )

        n += 1
    plt.show()

  ### Histos in 1-D array
  else:

    for n in range ( n_rows * n_cols ):

      x_1 = df[df["lymphoma_type"]==1][vars[n]]   # label "HL"
      x_2 = df[df["lymphoma_type"]==2][vars[n]]   # label "GZ"
      x_3 = df[df["lymphoma_type"]==3][vars[n]]   # label "PML"

      ax[n] . set_title ( vars[n][:40], size = 10 )

      if plot_style == "stacked":
        ax[n] . hist ( [x_1, x_2, x_3], bins = 20, stacked = True, color = colors, label = labels )

      elif plot_style == "overlapped":
        r_min = min ( x_1.min(), x_2.min(), x_3.min() )
        r_max = max ( x_1.max(), x_2.max(), x_3.max() )
        ax[n] . hist ( x_1, range = (r_min, r_max), bins = 20, alpha = 0.7, color = "dodgerblue" , label = "HL"  )
        ax[n] . hist ( x_2, range = (r_min, r_max), bins = 20, alpha = 0.7, color = "deeppink"   , label = "GZ"  )
        ax[n] . hist ( x_3, range = (r_min, r_max), bins = 20, alpha = 0.7, color = "forestgreen", label = "PML" )
        ax[n] . grid ( True, alpha = 0.3 )

      else:
        pass   # empty plots

      if plot_style in ["stacked", "overlapped"]:
        if n in legend_on_left:
          ax[n] . legend ( loc = "upper left" , fontsize = 8 )
        else:
          ax[n] . legend ( loc = "upper right", fontsize = 8 )
    plt.show()
