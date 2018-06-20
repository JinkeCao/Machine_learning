# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 09:59:50 2018

@author: jcao2014
"""
#import sys
# from xgboost import XGBClassifier, plot_importance
import pandas
# from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras import backend
from keras.models import Sequential 
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras import regularizers


# from sklearn.feature_extraction import DictVectorizer
# from matplotlib.pylab import rcParams
from sklearn.metrics import classification_report, roc_auc_score
from auc_callback import auc
import time
from plot_learning_curve import plot_learning_curve
import sys
#import 20180321_plot_tree
# rcParams['figure.figsize'] = 10,6




# input1= '20180323_kxjl_tags'
# df1 = pandas.read_table(input1)
# print(df2.shape)
# df3 = pandas.merge(df1, df2, on='dvc')
# print(df3.shape)
# x_train = df2[['source1', 'month', 'app_name_num', 'app_tag_num', 'load_time_num']]
# print(x_train.info())
# print(x_train['source1'].value_counts())
# dict_vec = DictVectorizer(sparse=False)
# X_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
# dict_vec.feature_names_
# time.sleep(100)

input1 = 'riskcontrol_tag_sample3.txt'
input0 = 'riskcontrol_tag_notsample3.txt'
df1 = pandas.read_table(input1)
df1['label'] = 1
df1_train = df1[df1['timestamp'] <= 1519488000]
df1_test = df1[df1['timestamp'] > 1519488000]


df0 = pandas.read_table(input0)
df0['label'] = 0
df0 = df0.sample(frac = 1)
df0_train = df0.head(df1_train.shape[0]*3)
df0_test = df0.tail(df1_test.shape[0]*3)

df_train = pandas.concat([df1_train, df0_train], ignore_index = True).sample(frac = 1)
df_test = pandas.concat([df1_test, df0_test], ignore_index = True).sample(frac = 1)
X_train, y_train = df_train['tags'], df_train['label']
X_test, y_test = df_test['tags'], df_test['label']



# df1['label'] = 1
# df2['label'] = 0
# df3 = pandas.concat([df1,df2],ignore_index = True) # unit  size = (64533,6)
# df4 = df3.sample(frac = 1) # 'frac = 1' represent the return ratio
# df5 = pandas.DataFrame()
# print(df4.shape)
# df6 = pandas.DataFrame()
# df6['time'] = df4['timestamp']
# print(df6.describe())
# df5 = df4[df4['timestamp'] > 1519488000 ]
# # 1519488000
# print((df5.shape))


# X_train, X_test, y_train, y_test = train_test_split(df4.tags,df4.label, test_size=0.2,random_state=231)
vec = CountVectorizer()
x_vec = vec.fit_transform(X_train)
ll = x_vec.toarray()
c = vec.get_feature_names()
x_train = pandas.DataFrame(data=ll, columns=c)
x_vec = vec.transform(X_test)
ll = x_vec.toarray()
x_test = pandas.DataFrame(data=ll, columns=c)
vocab_size = x_train.shape[1]
# print(vocab_size)
maxword = x_train.shape[1]





encoder = LabelEncoder()
encoded_train = encoder.fit_transform(y_train)
y_train = np_utils.to_categorical(encoded_train)
encoded_test = encoder.transform(y_test)
y_test = np_utils.to_categorical(encoded_test)
# print(y_test[0:11])

vocab_size = x_train.shape[1]
maxword = x_train.shape[1]

backend.clear_session()
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length = maxword))

model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same',activation = 'relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(32, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))

# model.add(LSTM(128, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(64, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(32))
# model.add(Dropout(0.2))

model.add(Dense(2, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = [auc])



print('x_train', x_train.shape, '\ny_train', y_train.shape,'\nx_test', x_test.shape, '\ny_test', y_test.shape)

print(model.summary())
print('training scores')
model.fit(x_train, 
          y_train, 
          epochs = 2, 
          batch_size = 32)
# callbacks = [
#     EarlyStoppingByLossVal(monitor='val_loss', value=0.0001, verbose=1),
#     # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
#     ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
# ]
# model.fit(x_train, y_train, batch_size=32, nb_epoch=32,
# #       shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
#       callbacks=callbacks)
loss, auc= model.evaluate(x_test, y_test, batch_size=128)
y_pred = model.predict(x_test)
# print(y_pred)
# print()
print('testing scores\n', '\nloss', loss, 'auc',auc, '\n')

print(classification_report(y_test[:, 1], list(map(lambda x: 1 if x[1] > x[0] else 0, y_pred))))
# print('testing scores', '\nloss', loss, '\nauc',auc)

# # model = XGBClassifier()
# # # # pa
# # # # plot_learning_curve(estimator=model, X=x_train, y=y_train, title='Learning curve on training sets', n_jobs=-1)

# # # model = XGBClassifier(n_jobs = -1, learning_rate=0.08, n_estimators=1000, max_depth=5, gamma=0, subsample=0.9, colsample_bytree=0.5)
# # # print(dir(model))
# # # time.sleep(100)
# # print('fitting..................')
# # model.fit(x_train, y_train)
# # print('preding..................')
# # y_pred = model.predict(x_test)
# # print('AUC = ', roc_auc_score(y_test, y_pred))
# # print(classification_report(y_test, y_pred))
# # plot_importance(model, max_num_features=20)


# from keras.models import Sequential 
# from keras.layers.embeddings import Embedding 
# from keras.preprocessing import sequence 
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv1D, MaxPooling1D
# from keras import backend
# backend.clear_session()

# model = Sequential()
# model.add(Embedding(vocab_size, 64, input_length = maxword))
# model.add(Conv1D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
# model.add(MaxPooling1D(pool_size = 2))
# model.add(Dropout(0.25))
# model.add(Conv1D(filters =64, kernel_size = 3, padding = 'same',activation = 'relu'))
# model.add(MaxPooling1D(pool_size = 2))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dense(1, activation = 'sigmoid'))
# # model.add(Dense(1, activation = 'softmax'))


# model.compile(loss = 'binary_crossentropy', 
#               optimizer = 'rmsprop', 
#               metrics = ['accuracy',auc])
# print(model.summary())
# model.fit(x_train, 
#           y_train, 
#           epochs = 2, 
#           batch_size = 128)
# with open('20180416_report', 'a') as f:
#     print('Scores on testing set:', file=f)
#     for i, j in zip(['loss', 'accuracy', 'auc'], model.evaluate(x_test, y_test, batch_size=128)):
#         print(i, j, file=f)
#     model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    
backend.clear_session()

