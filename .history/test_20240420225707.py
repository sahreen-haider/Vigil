import os
import numpy as np
import pandas as pd


data = pd.DataFrame(columns = ["no", 'binary'])

data[["no", "binary"]] = ["six", "bever"]
data[["no", "binary"]] = ["x", "never"]

print(data)