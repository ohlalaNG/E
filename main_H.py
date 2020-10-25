from typing import Any, Union

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from pandas.io.parsers import TextFileReader


#import visplots

h_data = pd.read_csv('file:///E:\Python%20projects\Datasets\melb_data.csv', index_col=0)
print(h_data.shape)
#pd.set_option('display.max_columns', 30)


#print(h_data.head().to_string())

#print(round(h_data.Price.describe(), 2))

#print(h_data.to_string())


#visplots.roompricebar()

#print(h_data.BuildingArea.describe())
