import os
import numpy as np
import pandas as pd


data = pd.DataFrame(columns = ["no", 'binary'])

data = data.append({"no": 77, "binary": True})
data = data.append({"no": 7, "binary": False})
print(data)