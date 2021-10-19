import numpy as np


def display_score ( scores: list , 
                    score_name : str = "accuracy" , 
                    model_name : str = "Some classifier" ):
  str_len = len ( model_name )
  print ( "+--" + "-" * str_len + "--+")
  print ( "|  {}  |" . format (model_name) )
  print ( "+--" + "-" * str_len + "--+")
  print ( "| {} : {}" . format ( score_name, scores ) )
  print ( "| mean : {:.1f}%" . format ( 100 * np.mean(scores) ) )
  print ( "| std  : {:.1f}%" . format ( 100 * np.std (scores) ) )
  print ( "+--- - -")
  