# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 09:59:50 2018

@author: jcao2014
"""
#import sys
from xgboost import XGBClassifier, plot_importance
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import classification_report, roc_auc_score
from auc_callback import auc
from time import sleep
from plot_learning_curve import plot_learning_curve
from matplotlib import pyplot
import seaborn
import datetime
from datetime import date
# from pandas.series.dt import date
# def dateparse (time_in_secs):    
#     return datetime.datetime.fromtimestamp(float(time_in_secs))

input1 = 'riskcontrol_tag_sample3.txt'
input2 = 'riskcontrol_tag_notsample3.txt'
df1 = pandas.read_table(input1)
df2 = pandas.read_table(input2)
# sr1 = pandas.series()
df1['label'] = 1
df2['label'] = 0
# print(dir(pandas))
# df1['datetime'] = pandas.to_datetime(df1['timestamp'])

for i in df1['timestamp']:
    dt = datetime.datetime.utcfromtimestamp(i)
    print(dir(dt))
    break
df1['day'] = df1.timestamp.apply(lambda x: datetime.datetime.utcfromtimestamp(x).year * 10000 + datetime.datetime.utcfromtimestamp(x).month * 100 + datetime.datetime.utcfromtimestamp(x).day)
df1['stringday'] = df1.day.apply(lambda x:str(x))


# frame['panduan'] = frame.city.apply(lambda x: 1 if 'ing' in x else 0)
print(df1.head(5), '\n')
df3 = df1['day']
print(df3.quantile(0.8), '\n')
# seaborn.distplot(df1['month'], norm_hist=False, kde=False)
seaborn.countplot(df1['day'])
# df3 = df1['stringday']
# print(df3.describe(), '\n')
# # seaborn.distplot(df1['month'], norm_hist=False, kde=False)
# seaborn.countplot(df1['stringday'])


# df1['date'] = df1['timestamp']

# print(df3.describe())

# seaborn.distplot(df1['month'])
# seaborn.plt.show()
# df3 = df1.month
# print(df3.describe())

# df3 = pandas.concat([df1,df2],ignore_index = True) # unit  size = (64533,6)
# df4 = df3.sample(frac = 1) # 'frac = 1' represent the return ratio


# X_train, X_test, y_train, y_test = train_test_split(df4.tags,df4.label, test_size=0.2,random_state=231)
# vec = CountVectorizer()
# x_vec = vec.fit_transform(X_train)
# ll = x_vec.toarray()
# c = vec.get_feature_names()
# x_train = pandas.DataFrame(data=ll, columns=c)
# x_vec = vec.transform(X_test)
# ll = x_vec.toarray()
# x_test = pandas.DataFrame(data=ll, columns=c)
# vocab_size = x_train.shape[1]
# maxword = x_train.shape[1]

# model = XGBClassifier(nthread=4, learning_rate=0.08, n_estimators=50, max_depth=5, gamma=0, subsample=0.9, colsample_bytree=0.5)
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# plot_importance(model, max_num_features=20)
# print('AUC = ', roc_auc_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


