
############################## Store Item Demand Forecasting #############################

############################## Libraries and Utilities ###################################
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate,\
    validation_curve, train_test_split
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer

from helpers.eda import *
from helpers.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

######################### Loading the datas ########################

train = pd.read_csv("datasets/demand_forecasting/train.csv", parse_dates=["date"])
test = pd.read_csv("datasets/demand_forecasting/test.csv", parse_dates=["date"])
sample_sub = pd.read_csv("datasets/demand_forecasting/train.csv", parse_dates=["date"])
df = pd.concat([train, test], sort=False)

#####################  Exploratory Data Analysis ###########################

check_df(train)
check_df(test)

train["date"].min(), train["date"].max(), test["date"].min(), test["date"].max()

# How is the sales distribution?
df[["sales"]].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T

# How many unique stores?
df["store"].nunique()

# How many of each product have been sold?
df["item"].value_counts()

# Top 5 most expensive sales
a = df["sales"].sort_values(ascending=False).head().index
for index in a:
    print(df[df.index == index])

# Are there an equal number of unique items in each store??
df.groupby(["store"])["item"].nunique()

# Sales statistics in store breakdown
df.groupby("store").agg({"sales": ["sum", "mean", "median", "std"]})
df.groupby("item").agg({"sales": ["sum", "mean", "median", "std"]})

sns.lineplot(x="date",y="sales", legend="full",data=train)
plt.show()

df.plot(x="date", y="sales", alpha=0.5)
plt.show()

################################### FEATURE ENGINEERING ##################################

######################### Date Features ########################

def create_date_features(dataframe):
    dataframe['month'] = dataframe.date.dt.month
    dataframe['day_of_month'] = dataframe.date.dt.day
    dataframe['day_of_year'] = dataframe.date.dt.dayofyear
    dataframe['week_of_year'] = dataframe.date.dt.weekofyear
    dataframe['day_of_week'] = dataframe.date.dt.dayofweek
    dataframe['year'] = dataframe.date.dt.year
    dataframe["is_wknd"] = dataframe.date.dt.weekday // 4
    dataframe['is_month_start'] = dataframe.date.dt.is_month_start.astype(int)
    dataframe['is_month_end'] = dataframe.date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)

check_df(df)

df.groupby(["store", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})

######################### Random Noise ########################

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

a = np.random.normal(scale=1.6, size=(len(df)))

a = pd.DataFrame(a)
a.quantile([0, 0.1, 0.15, 0.25, 0.50, 0.75, 0.8, 0.84, 0.9, 0.95, 0.99, 1])

sns.distplot(a)
plt.show()

######################### Lag/Shifted Features ########################

df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

check_df(df)

pd.DataFrame({"sales": df["sales"].values[0:10],
              "lag1": df["sales"].shift(1).values[0:10],
              "lag2": df["sales"].shift(2).values[0:10],
              "lag3": df["sales"].shift(3).values[0:10],
              "lag4": df["sales"].shift(4).values[0:10]})


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

check_df(df)


######################### Rolling Mean Features ########################


pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].rolling(window=5).mean().values[0:10]})


pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10]})


def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [365, 546])
df.tail()



######################### Exponentially Weighted Mean Features ########################

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm01": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})


def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)

check_df(df)

######################### One-Hot Encoding ########################

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])

######################### Converting sales to log(1+sales) ########################

df['sales'] = np.log1p(df["sales"].values)
check_df(df)

###################################################### Model #####################################################

######################### Custom Cost Function ########################

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

######################### Time-Based Validation Sets ########################

# Train set until the beginning of 2017 (until the end of 2016).
train = df.loc[(df["date"] < "2017-01-01"), :]

# Validation set for the first three months of 2017
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, Y_train.shape, Y_val.shape, X_val.shape

################################ Model ################################

lgb_model = LGBMRegressor(random_state=1).fit(X_train, Y_train)

# Train Error
print("Train SMAPE:", "{:,.4f}".format(smape(np.expm1(Y_train), np.expm1(lgb_model.predict(X_train)))), "\n")

# Test Error
print("Test SMAPE:", "{:,.4f}".format(smape(np.expm1(Y_val), np.expm1(lgb_model.predict(X_val)))), "\n")


tscv = TimeSeriesSplit(n_splits=3)
lgb_model = LGBMRegressor(random_state=1)

lightgbm_params = {"learning_rate": [0.01, 0.001],
                   "n_estimators": [100, 1500],
                   "colsample_bytree": [0.8, 1],
                   "max_depth":[10, 14]}

rf_best_grid = GridSearchCV(lgb_model,
                            lightgbm_params,
                            cv=tscv,
                            scoring=make_scorer(smape),
                            n_jobs=-1,
                            verbose=True).fit(X_train, Y_train)

lgb_final = lgb_model.set_params(**rf_best_grid.best_params_,
                               random_state=1).fit(X_train, Y_train)

print("Train SMAPE:", "{:,.4f}".format(smape(np.expm1(Y_train), np.expm1(lgb_final.predict(X_train)))), "\n")
print("Train SMAPE:", "{:,.4f}".format(smape(np.expm1(Y_val), np.expm1(lgb_final.predict(X_val)))), "\n")

def plot_importance(model, features, num=25):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(15, 15))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()

plot_importance(lgb_final, X_train)

################################# Analyzing Model Complexity with Learning Curves ################################

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=tscv):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = smape(np.expm1(train))
    mean_test_score = smape(np.expm1(test))

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

lgb_model = LGBMRegressor(random_state=17)


lightgbm_params = [["learning_rate", [0.01, 0.001]],
                   ["n_estimators", [100, 1500]],
                   ["colsample_bytree", [0.8, 1]],
                   ["max_dept", [10, 14]]]

for i in range(len(lightgbm_params)):
    val_curve_params(lgb_model, X_train, Y_train, lightgbm_params[i][0], lightgbm_params[i][1], "neg_root_mean_squared_error")

######################### LightGBM Model ########################

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}


# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))


######################### Feature Importance ########################


def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30)
plot_lgb_importances(model, num=30, plot=True)

######################### Final Model ########################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}


# LightGBM dataset
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

test_preds = model.predict(X_test, num_iteration=model.best_iteration)


submission_df = test.loc[:, ['id', 'sales']]
submission_df['sales'] = np.expm1(test_preds)
submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv('submission_demand.csv', index=False)
submission_df.head(20)