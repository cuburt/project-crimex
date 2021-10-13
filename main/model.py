import os
import json
import pandas as pd
import pickle

from sklearn import preprocessing, gaussian_process
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, ConstantKernel, RBF
from datetime import datetime
from scipy.optimize import minimize,least_squares
import math
import os
import pandas as pd
import numpy as np
import json
import logging

import time
import datetime

_logger = logging.getLogger(__name__)

token = 'pk.eyJ1IjoiY3VidXJ0IiwiYSI6ImNrdWtwbnc3eTB3NHQyeHJ2c2Y2M20wc2UifQ.Uz4NImceExllY5zXyFvvug'
with open(os.path.join(os.path.dirname(__file__), "static/regions.json"),'r') as f:
    regions = json.load(f)

psgc_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "static/clean-psgc.csv"),
                   dtype={"code": str})

crime_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "static/crime_rate_dataset.csv"))

crime_df['PERIOD'] = crime_df[['MONTH', 'YEAR']].astype(str).apply(lambda x: ''.join(x), axis=1)
crime_df['DATETIME'] = pd.to_datetime(crime_df.YEAR.astype(str) + '/' + crime_df.MONTH.astype(str))
crime_df.set_index('DATETIME', inplace=True)



reg_df = psgc_df.loc[psgc_df['interlevel']=='Reg']
prov_df = psgc_df.loc[psgc_df['interlevel']=='Prov']
city_df = psgc_df.loc[psgc_df['interlevel']=='City']
mun_df = psgc_df.loc[psgc_df['interlevel']=='Mun']
bgy_df = psgc_df.loc[psgc_df['interlevel']=='Bgy']


def filter_dataset(selected_months, selected_years, filters, level, crime):

    if selected_months:
        if selected_months[0] != selected_months[1]:
            selected_months = [i for i in range(selected_months[0],selected_months[1]+1)]
            filtered_df = crime_df.loc[crime_df.YEAR.isin(selected_years)]
            filtered_df = filtered_df.loc[filtered_df.M.isin(selected_months)]
            for code in crime_df['CODE'].unique():
                temp_df = filtered_df.loc[filtered_df.CODE == code]
                volume = temp_df[crime].sum(axis=1)
                temp_df['crime_volume'] = volume
                rate = (temp_df['crime_volume'] / temp_df['POPULATION']) * 100000
                temp_df['crime_rate'] = rate
                filtered_df.loc[filtered_df.CODE == code, 'CRIME_VOLUME'] = volume
                filtered_df.loc[filtered_df.CODE == code, 'CRIME_RATE'] = rate
                filtered_df.loc[filtered_df.CODE == code, ['AVERAGE_MONTHLY_CRIME_RATE']] = temp_df['crime_rate'].sum() / len(selected_months)
        elif selected_months[0] == selected_months[1]:
            selected_months = [i for i in range(selected_months[0], selected_months[1] + 1)]
            filtered_df = crime_df.loc[crime_df.YEAR.isin(selected_years)]
            filtered_df = filtered_df.loc[filtered_df.M.isin(selected_months)]
            for code in crime_df['CODE'].unique():
                temp_df = filtered_df.loc[filtered_df.CODE == code]
                volume = temp_df[crime].sum(axis=1)
                temp_df['crime_volume'] = volume
                rate = (temp_df['crime_volume'] / temp_df['POPULATION']) * 100000
                temp_df['crime_rate'] = rate
                filtered_df.loc[filtered_df.CODE == code, 'CRIME_VOLUME'] = volume
                filtered_df.loc[filtered_df.CODE == code, 'CRIME_RATE'] = rate
                filtered_df.loc[filtered_df.CODE == code, ['AVERAGE_MONTHLY_CRIME_RATE']] = temp_df['crime_rate']
    else:
        selected_months = [i for i in crime_df['M'].unique()]
        selected_years = [i for i in crime_df['YEAR'].unique()]
        filtered_df = crime_df.loc[crime_df.YEAR.isin(selected_years)]
        filtered_df = filtered_df.loc[filtered_df.M.isin(selected_months)]
        for code in crime_df['CODE'].unique():
            temp_df = filtered_df.loc[filtered_df.CODE == code]
            volume = temp_df[crime].sum(axis=1)
            temp_df['crime_volume'] = volume
            rate = (temp_df['crime_volume'] / temp_df['POPULATION']) * 100000
            temp_df['crime_rate'] = rate
            filtered_df.loc[filtered_df.CODE == code, 'CRIME_VOLUME'] = volume
            filtered_df.loc[filtered_df.CODE == code, 'CRIME_RATE'] = rate
            filtered_df.loc[filtered_df.CODE == code, ['AVERAGE_MONTHLY_CRIME_RATE']] = temp_df['crime_rate']

    return filtered_df

#cubic spline function: for interpolation
def upsample(df):
    proj_list = [proj for proj in df['Name of Project'].values]
    project_list = []
    for project in set(proj_list):
        rec_df = df.loc[df['Name of Project'] == project]
        order = (rec_df.shape[0]) - 1
        if order >= 5: order = 5
        itpd_df = rec_df.resample("D").interpolate(method='spline', order=order)
        itpd_df = itpd_df.fillna(method='pad')
        itpd_df['Contract Amount'] = itpd_df['Contract Amount'].astype(float)
        itpd_df['No. of Days'] = itpd_df['No. of Days'].astype(int)
        project_list.append(itpd_df)
    return project_list

def trust_region_optimizer(obj_func, initial_theta, bounds):
    trust_region_method = least_squares(1/obj_func,initial_theta,bounds,method='trf')
    return (trust_region_method.x,trust_region_method.fun)

def define_model():
    k0 = WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))
    k1 = ConstantKernel(constant_value=2)* \
         ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35,45))
    k2 = ConstantKernel(constant_value=10, constant_value_bounds=(1e-2, 1e3))* \
         RBF(length_scale=100.0, length_scale_bounds=(1, 1e4))
    kernel_1 = k0 + k1 + k2
    linear_model = gaussian_process.GaussianProcessRegressor(kernel=kernel_1, n_restarts_optimizer=10, alpha=0.5)
#     linear_model = gaussian_process.GaussianProcessRegressor(optimizer = trust_region_optimizer, n_restarts_optimizer=10, alpha=0.5)
    return linear_model

def score_model(X, y_test, y_pred):
    ssr = sum((y_test-y_pred)**2)
    mse = 1/len(y_pred)*ssr
    rmse = np.sqrt(mse)
    sst = ((y_test-np.mean(y_test))**2)
    r2 = 1 - (ssr/sst)
    adj_r2 = 1-(1-r2)*(len(y_test)-1)/(len(y_test)-(len(X.columns))-1)
    return [np.mean(adj_r2),rmse]

def train_model(X_lately, X, y_lately, y, max_iter, max_perc, test_size):
    for r in range(max_iter):
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=r)
        model = define_model().fit(X_train, y_train)
        # model_score = model.score(X_test, y_test)
        # if model_score >= max_perc/100:
        #     y_test_pred = model.predict(X_test)
        #     y_pred = model.predict(X_lately)
        y_test_pred = model.predict(X_test)
        val_model_score = score_model(X_test, y_test, y_test_pred)
        print(val_model_score)
        if val_model_score[1] <= (max_perc/100):
            try:
                mainmodel = define_model().fit(X, y)
                y_pred = mainmodel.predict(X_lately)
                model_score = score_model(X_lately, y_lately, y_pred)
                return [X_train, X_test, y_train, y_test, y_pred, model_score[0],model_score[1]]
            except Exception as e:
                print(str(e))
                break

class CustomWalkForward:
    def __init__(self, test_size, gap):
        self.test_size = test_size
        self.train_size = 1 - test_size
        self.gap = gap

    def split(self,df):
        X = df
        n = len(X)
        folds = int(math.ceil(n/(n*self.test_size)))
        q = int(n/folds)
        res = n%folds
        for k in range(1,folds+1):
            train_range = int((q*k)*self.train_size)
            if k == folds: train_range = int((q*k)*self.train_size+res)
            train_set = X.head(train_range-self.gap)
            test_set = X[train_range:train_range+int((q*k)*self.test_size)]
            yield np.array(train_set.index),np.array(test_set.index)

def forecast(df, col):
    max_iter = 5000
    max_perc = 60
    test_size = 0.2
    tscv = CustomWalkForward(test_size=0.3, gap=0)
    # itpdf = pd.concat(project_list)
    itpdf = df[col]
    ave_rmse = []
    ave_score = []
    for train_index, test_index in tscv.split(itpdf):
        print('TRAIN: ', train_index, 'TEST: ', test_index)
        cv_df = df[[col]]
        forecast_col = col
        X = cv_df
        scaler = preprocessing.MinMaxScaler()
        scaled_X = scaler.fit_transform(X)
        data = {col: scaled_X[:, 0]}
        X = pd.DataFrame(data=data, index=X.index)
        X, X_lately = X.loc[train_index], X.loc[test_index]
        # X, X_lately = cv_df.loc[train_index], cv_df.loc[test_index]
        #cv_df, the y values of train_index shifted forward:
        cv_df['train'] = cv_df[forecast_col].head(len(train_index) + len(test_index)).shift(-(len(test_index)))

        cv_df.dropna(inplace=True)
        cv_df['test'] = cv_df[forecast_col].head(len(test_index))
        cv_df['test'].dropna(inplace=True)
        #y is the y axis of the train_index, just shifted forward.
        y = np.array(cv_df['train'])
        y_lately = np.array(cv_df['test'])
        y_lately = y_lately[np.logical_not(np.isnan(y_lately))]
        print('training...')
        full_set = train_model(X_lately, X, y_lately, y, max_iter, max_perc, test_size)
        print(full_set)
        y_pred = full_set[4]
        ave_score.append(full_set[5])
        ave_rmse.append(full_set[6])
        cv_df['Forecast'] = np.nan

        last_month = cv_df.iloc[-1].name
        last_unix = last_month.timestamp()
        one_month = 2592000
        next_unix = last_unix + one_month

        for i in y_pred:
            next_month = datetime.datetime.fromtimestamp(next_unix)
            next_unix += one_month
            cv_df.loc[next_month] = [np.nan for _ in range(len(cv_df.columns) - 1)] + [i]

        cv_df['train'] = cv_df['train'].shift(len(test_index))
        cv_df['Forecast'] = np.nan

        for i in y_pred:
            next_month = datetime.datetime.fromtimestamp(next_unix)
            next_unix += one_month
            cv_df.loc[next_month] = [np.nan for _ in range(len(cv_df.columns) - 1)] + [i]

    for lead in cv_df['train'].values:
        for val in cv_df[col].values:
            if lead == val:
                cv_df['train'].loc[cv_df['train']==lead] = np.nan
    print('SCORE: ', np.mean(ave_score), '\nRMSE: ', np.mean(ave_rmse))
    return [cv_df, cv_df.reset_index(), np.mean(ave_score), np.mean(ave_rmse)]

#saving forecast to pickle
def init_dataset():
    crime = ['MUR', 'HOM', 'PHY_INJ', 'RAPE', 'ROB', 'THEFT', 'MV', 'MC', 'CATTLE_RUSTLING', 'RIRT_HOM', 'RIRT_PHY_INJ',
             'RIRT_DTP', 'VIOL_OF_SPECIAL_LAWS', 'OTHER_NON_INDEX_CRIMES']
    selected_months = [i for i in crime_df['M'].unique()]
    selected_years = [i for i in crime_df['YEAR'].unique()]
    filtered_df = crime_df.loc[crime_df.YEAR.isin(selected_years)]
    filtered_df = filtered_df.loc[filtered_df.M.isin(selected_months)]
    for code in crime_df['CODE'].unique():
        temp_df = filtered_df.loc[filtered_df.CODE == code]
        volume = temp_df[crime].sum(axis=1)
        temp_df['crime_volume'] = volume
        rate = (temp_df['crime_volume'] / temp_df['POPULATION']) * 100000
        temp_df['crime_rate'] = rate
        filtered_df.loc[filtered_df.CODE == code, 'CRIME_VOLUME'] = volume
        filtered_df.loc[filtered_df.CODE == code, 'CRIME_RATE'] = rate
        filtered_df.loc[filtered_df.CODE == code, ['AVERAGE_MONTHLY_CRIME_RATE']] = temp_df['crime_rate']

    return filtered_df

filtered_df = init_dataset()
for code in filtered_df['CODE'].unique():
#     print(filtered_df.loc[filtered_df.CODE == code])
#     df = upsample(filtered_df.loc[filtered_df.CODE == code])
    with open(os.path.join(os.path.dirname(__file__), "static/" + str(code) + ".pickle"), 'rb') as data:
        dataset = pickle.load(data)
    if dataset:
        df = dataset
    else:
        df = forecast(filtered_df.loc[filtered_df.CODE == code], 'CRIME_RATE')[0]
        with open(os.path.join(os.path.dirname(__file__), "static/"+str(code)+".pickle"), 'wb') as output:
            pickle.dump(df, output)