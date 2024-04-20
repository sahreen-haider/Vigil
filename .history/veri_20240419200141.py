import numpy as np
import pandas as pd
import datetime

df = pd.read_csv("dataset/recorded_encodings/recorded_encode.csv")
df[["title", "encoding", "timestamp"]] = [["fgjfg", "dfgg", "345"]]

print(df["encoding"].iloc[-1])