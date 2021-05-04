import math

import pandas as pd
import numpy as np
from pandasgui import show
from os import listdir


# one file training data
training_df = pd.DataFrame
#training_df = pd.read_csv("experiment1/train/2018-09-18T20-03-15-500000_1_data_wide.csv")



#All file training data
li = []
for file in listdir("./experiment1/train"):
    df = pd.read_csv("./experiment1/train/" + file)
    li.append(df)
training_df = pd.concat(li, ignore_index=True)

# single file test data
test_df =pd.read_csv("experiment1/test/2018-09-24T23-22-15-500000_9_data_wide.csv")


#Nearest Neighbor

# methods for finding the lowest ED for a test row (iterating all training rows)
edge_columns = ["edge_1","edge_2","edge_3","edge_8","edge_9","edge_10","edge_11","edge_12","edge_13"]
xy_columns = ["realx","realy"]

def euclidean_distance(training_row, test_row, columns):
    inner_value = 0
    for k in columns:
        inner_value += (training_row[k] - test_row[k]) **2
    return math.sqrt(inner_value)

def get_lowest_ED(test_row):
    resultframe = training_df
    resultframe["ED"] = resultframe.apply(lambda row: euclidean_distance(row, test_row, edge_columns), axis = 1)
    resultframe.sort_values("ED", inplace=True)
    return euclidean_distance(resultframe.iloc[0], test_row, xy_columns)



results_df = pd.DataFrame

#appending or new df for each test row add coordinats for traing rowwith lowest ED
# calculate distance between guessed x,y and observed x,y and ad this as column
test_df["distance_error"] = test_df.apply(lambda row: get_lowest_ED(row), axis = 1)

# do statistics on error column
print("mean: " + str(test_df["distance_error"].mean()))
print("SD: " + str(test_df["distance_error"].std()))
#show(training_df, settings={"block": True})