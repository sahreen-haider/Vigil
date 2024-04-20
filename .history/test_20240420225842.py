import os
import numpy as np
import pandas as pd


data = pd.DataFrame(columns = ["no", 'binary'])

data[["no", "binary"]].iloc[0] = [["six", "bever"]]
data[["no", "binary"]].iloc[1] = [["x", "never"]]

print(data)