
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
color = sns.color_palette()

# %matplotlib inline


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

for col in df_all.columns:
  if df_all[col].dtypes != 'object':
    upper_limit = np.percentile(df_all[col].dropna(), 99)
    lower_limit = np.percentile(df_all[col].dropna(), 1)

    df_all.loc[(df_all[col] > upper_limit), col] = np.NaN
    df_all.loc[(df_all[col] < lower_limit), col] = np.NaN

df_all.f1 = pd.to_datetime(df_all.f1)

# Add month-year-count
month_year = (df_all.f1.dt.month + df_all.f1.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.f1.dt.weekofyear + df_all.f1.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['year'] = df_all.f1.dt.year
df_all['month'] = df_all.f1.dt.month
df_all['dow'] = df_all.f1.dt.dayofweek

df_all.drop(['f1'], axis=1, inplace=True)

num_train = len(train_df)

df_all.f7[df_all.f7 < 1900] = np.NaN

obj = df_all.select_dtypes(include=['object']).columns

obj = obj.drop('f11')

for c in obj:
    df_all[c] = pd.factorize(df_all[c])[0]

X_train_all = df_all[:num_train]
X_test = df_all[num_train:]

X_train_all['prediction'] = ylog_train_all

data_inv = X_train_all[X_train_all.f11 == 'Investment'].drop('f11', axis = 1)
data_own = X_train_all[X_train_all.f11 == 'OwnerOccupier'].drop('f11', axis = 1)
test_inv = X_test[X_test.f11 == 'Investment'].drop('f11', axis = 1)
test_own = X_test[X_test.f11 == 'OwnerOccupier'].drop('f11', axis = 1)

train_inv, val_inv = data_inv[int(len(data_inv)*0.2):], data_inv[:int(len(data_inv)*0.2)]
train_own, val_own = data_own[int(len(data_own)*0.2):], data_own[:int(len(data_own)*0.2)]


df_columns = data_inv.columns[:-1]

dtrain_all_inv = xgb.DMatrix(data_inv.drop('prediction', axis = 1), data_inv.prediction, feature_names=df_columns)
dtrain_inv = xgb.DMatrix(train_inv.drop('prediction', axis = 1), train_inv.prediction, feature_names=df_columns)
dval_inv = xgb.DMatrix(val_inv.drop('prediction', axis = 1), val_inv.prediction, feature_names=df_columns)
dtest_inv = xgb.DMatrix(test_inv, feature_names=df_columns)

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

model_inv = xgb.train(dict(xgb_params, silent=0), dtrain_all_inv, num_boost_round=num_boost_round_inv)


dtrain_all_own = xgb.DMatrix(data_own.drop('prediction', axis = 1), data_own.prediction, feature_names=df_columns)
dtrain_own = xgb.DMatrix(train_own.drop('prediction', axis = 1), train_own.prediction, feature_names=df_columns)
dval_own = xgb.DMatrix(val_own.drop('prediction', axis = 1), val_own.prediction, feature_names=df_columns)
dtest_own = xgb.DMatrix(test_own, feature_names=df_columns)

xgb_params = {
    'eta': 0.05,
    'max_depth':5,
    'subsample': 0.6,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}


model_own = xgb.train(dict(xgb_params, silent=0), dtrain_all_own, num_boost_round=num_boost_round_own)


ylog_pred_inv = model_inv.predict(dtest_inv)
y_pred_inv = np.exp(ylog_pred_inv) - 1
ylog_pred_own = model_own.predict(dtest_own)
y_pred_own = np.exp(ylog_pred_own) - 1

df_sub_inv = pd.DataFrame({'id': test_inv.index, 'prediction': y_pred_inv})
df_sub_own = pd.DataFrame({'id': test_own.index, 'prediction': y_pred_own})

df_sub = pd.concat([df_sub_inv, df_sub_own]).sort_values(by = 'id')

df_sub['id'] += 1

df_sub.to_csv('Кучеренко_Александр.csv', index = False)

