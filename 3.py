import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


h_data = pd.read_csv('file:///E:\Python%20projects\Datasets\melb_data.csv', index_col=0)

y = h_data.Price
X0 = h_data.drop(['Price'], axis=1)

#cols_to_drop = [col for col in X0.columns if X0[col].nunique()>=10 and X0[col].dtype == 'object']
#print(cols_to_drop)
#X0=X0.drop(columns=cols_to_drop)
#print(X0.shape)

cols1 = [col for col in X0.columns if X0[col].dtype in ['int64', 'float64']]
cols2 = [col for col in X0.columns if X0[col].nunique()<=10 and X0[col].dtype == 'object']
X=X0[cols1+cols2]

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
#print(X.head().to_string())

numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, cols1),
        ('cat', categorical_transformer, cols2)
    ])

model = XGBRegressor(n_estimators=1000, learning_rate=0.05)


val_X_eval=val_X.copy()
eval_set_pipe = Pipeline(steps = [('preprocessor', preprocessor)])
val_X_eval = eval_set_pipe.fit(train_X, train_y).transform(val_X_eval)
pip1 = Pipeline(steps = [('preprocessor', preprocessor), ('model', model)])

pip1.fit(train_X, train_y, model__early_stopping_rounds=5, model__eval_set=[(val_X_eval, val_y)])

predicted_y = pip1.predict(val_X)

print(mean_absolute_error(val_y, predicted_y))