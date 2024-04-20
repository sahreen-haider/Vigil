import os
import numpy as np
import pandas as pd


data = pd.DataFrame(columns = ["no", 'binary'])

data.iloc[0] = [["six", "appeal"]]
data.iloc[1] = [['try', "never"]]
print(data)