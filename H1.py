from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

h_data = pd.read_csv('file:///E:\Python%20projects\Datasets\melb_data.csv', index_col=0)
y = h_data.Price
h_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = h_data[h_features]


def pre_dt(m, n):
    model1 = DecisionTreeRegressor(random_state=0)
    model1.fit(m, n)
    print(model1.predict(m))
    return mean_absolute_error(n, model1.predict(m))



def pre_dt2(k, m, n):
    train_m, val_m, train_n, val_n = train_test_split(m, n, random_state=0)
    model2 = DecisionTreeRegressor(max_leaf_nodes=k, random_state=0)
    model2.fit(train_m, train_n)
    # print(model2.predict(val_m))
    return mean_absolute_error(val_n, model2.predict(val_m))


def findmaxleafnodes(k, m, n):
    a = 2
    b = pre_dt2(2, m, n)
    maxi = 2
    while a <= k:
        c = pre_dt2(a, m, n)
        if c < b:
            b = c
            maxi = a
        a = a + 1
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (maxi, b))

#findmaxleafnodes(650,X,y)
