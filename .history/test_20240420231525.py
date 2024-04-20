import os
import numpy as np
import pandas as pd


data = pd.DataFrame(columns = ["no", 'binary'])

data.loc(len(data)) = ["88", "09"]
print(data)