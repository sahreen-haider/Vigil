import os
import numpy as np
import pandas as pd


data = pd.DataFrame([33, 66], columns = ["no", 'binary'])

new_record = [10, 11]
data = data.append(new_record)
print(data)