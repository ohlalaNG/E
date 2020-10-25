import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats

h_data = pd.read_csv('file:///E:\Python%20projects\Datasets\melb_data.csv', index_col=0)

def roompricebar():
    plt.figure(figsize=(8,6))
    sb.barplot(x=h_data['Rooms'], y=h_data['Price'])
    plt.show()

def areapriceline():
    df = pd.DataFrame(h_data.BuildingArea)
    df['Price'] = h_data.Price
    df.dropna(inplace=True)
    df['z_score'] = stats.zscore(df.BuildingArea)
    df1 = df.loc[df['z_score'].abs() <= 3]
    plt.figure(figsize=(20, 6))
    sb.lineplot(x=df1['BuildingArea'], y=df1['Price'])
    plt.show()
