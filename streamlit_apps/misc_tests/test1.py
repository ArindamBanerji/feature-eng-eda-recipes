import numpy as np
import pandas as pd
import os
cwd = os.getcwd()
# print ("current dir", cwd)

# Print the current working directory
print ("Current working directory: {0}".format(cwd))

# Print the type of the returned object
print ("os.getcwd() returns an object of type: {0}".format(type(cwd)))


df = pd.read_csv ( '../datasets/IPL_B2B_Dataset.csv' )
print ("read file")

ax = df.head(5)

# print (ax)