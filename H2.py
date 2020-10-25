from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

h_data = pd.read_csv('file:///E:\Python%20projects\Datasets\melb_data.csv', index_col=0)
y = h_data.Price
h_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X=h_data[h_features]
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

model3 =RandomForestRegressor(n_estimators=10, random_state=0)
model3.fit(train_X, train_y)
#missing values included
#print(mean_absolute_error(val_y, model3.predict(val_X)))

cols_with_missing = [col for col in train_X.columns
                     if train_X[col].isnull().any()]

train_X1 = train_X.copy()
val_X1 = val_X.copy()

for col in cols_with_missing:
    train_X1[col + '_missing'] = train_X1[col].isnull()
    val_X1[col + '_missing'] = val_X1[col].isnull()

imputer1 = SimpleImputer()
imputed_train_X = pd.DataFrame(imputer1.fit_transform(train_X1))
imputed_val_X = pd.DataFrame(imputer1.transform(val_X1))
imputed_train_X.columns = train_X1.columns
imputed_val_X.columns = val_X1.columns

model3.fit(imputed_train_X, train_y)

predicted_y = model3.predict(imputed_val_X)

print(mean_absolute_error(val_y, predicted_y))

#print(train_X.shape)
#print(imputed_train_X.shape)

#print(imputed_train_X.head())
