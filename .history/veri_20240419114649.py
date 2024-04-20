import numpy as np
import pandas as pd

df = pd.DataFrame(columns = ["title", "encoding", "timestamp"])

df.to_csv("dataset/recorded_encodings/recorded_encode.csv")