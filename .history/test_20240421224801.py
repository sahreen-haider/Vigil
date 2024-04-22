# import os
# import numpy as np
# import pandas as pd


# data = pd.DataFrame(columns = ["no", 'binary'])

# data.loc[len(data)] = ["88", "09"]
# data.loc[len(data)] = ["99", "fg"]
# print(data)


# Create an initial dictionary
dict_example = {'a': 1, 'b': 2}

# Update the dictionary with new key-value pairs
dict_example.update({'c': 3})

# Print the updated dictionary
print(dict_example)
print(dict_example.popitem()[1])