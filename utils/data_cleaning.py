import numpy as np
import pandas as pd

from datetime import datetime


def data_cleaning ( db : pd.DataFrame ,
                    col_name    : str , 
                    target_type : str , 
                    inplace : bool = False ) -> pd.DataFrame :
  if not isinstance ( col_name, str ):
    raise TypeError ( "The column name should be a string, %s passed" % type(col_name) )
  values = list ( db[col_name] )

  if target_type in ( 'int', 'float' ):
    for i_ in range ( len(values) ):
      if isinstance ( values[i_], str ):
        values[i_] = values[i_] . replace ( ",", "." )
        if target_type == 'int'   : values[i_] = int   ( values[i_] )
        if target_type == 'float' : values[i_] = float ( values[i_] )
  elif target_type == 'date':
    for i_ in range ( len (values) ):
      if isinstance ( values[i_], str ):
        day, month, year = str ( values[i_] ) . split ( "/" )
        values[i_] = datetime ( int(year), int(month), int(day) )
  elif target_type == 'str':
    pass
  else:
    raise ValueError ("Target type not implemented, values allowed: ['int', 'float', 'date', 'str']")

  if inplace:
    db[col_name] = values
    return db
  else:
    new_db = db.copy()
    new_db[col_name] = values
    return new_db



if __name__ == "__main__":
  db_test = pd.DataFrame (columns = ["A"])   # test database
  db_test["A"] = [ "1,0" for i in range(10) ]         # invalid values
  db_test["B"] = [ "17/07/2021" for i in range(10) ]  # invalid values

  print("+-------------------------------+")
  print("|   Database before cleaning:   |")
  print("+-------------------------------+")
  print(db_test)

  data_cleaning (db_test, col_name = "A", target_type = "float", inplace = True)
  data_cleaning (db_test, col_name = "B", target_type = "date" , inplace = True)

  print("\n+-------------------------------+")
  print("|   Database after cleaning:    |")
  print("+-------------------------------+")
  print(db_test)
