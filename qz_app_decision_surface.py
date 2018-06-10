print(__doc__)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pandas import read_table, DataFrame
from sklearn import clone
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
matplotlib.use('TkAgg')


# Parameters
n_classes = 3
n_estimators = 30
cmap = plt.cm.RdYlBu
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
RANDOM_SEED = 13  # fix the seed on each iteration

# Load data
from json import loads
import numpy as np



def list2str(x):
    appstr = ''
    if x != 'null' and x!='0':
        x = loads(x)
        for i in x:
            if i['load_info']!=None:
                n = len(i['load_info'])
                for j in range(n + 1):
                    appstr += (i['app_name'] + ' ')
    return appstr



df1 = read_table('/run/shm/jupyter/yldong3/data/qianzhan0514.txt', header=None, names=['dvc', 'collect_day', 'app_info','type', 'source1', 'source2'])
print(df1.shape)
df1.fillna(int(0), inplace=True)
df1.drop_duplicates(["dvc"], inplace=True)
# df1 = df1[(df1['source2']=='loan_time')]
# df1 = df1[(df1['source1']=='ime_app_install') | (df1['source1']=='sdk_log_install')]
# df1 = df1[(df1['source1']=='ime_app_install') | (df1['source1']=='sdk_log_install')]
df1['type'], df1['app_info'] = df1.type.apply(int), df1.app_info.apply(list2str)

vec = CountVectorizer()
x_vec1 = vec.fit_transform(df1['app_info'])
l1 = x_vec1.toarray()
c = vec.get_feature_names()
x =  DataFrame(data=l1, columns=c)
x.fillna(0, inplace=True)
print('X',x.shape)


plot_idx = 1

models = [DecisionTreeClassifier(max_depth=None),
          RandomForestClassifier(n_estimators=n_estimators),
          ExtraTreesClassifier(n_estimators=n_estimators),
          AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                             n_estimators=n_estimators)]

# for pair in ( ['钱站', '铜掌柜理财'],['钱站', '拉卡拉钱包'], ['钱站', '闪银奇异']):
for pair in (['钱站','银闪付'],['钱站','拉卡拉钱包'],['钱站','闪银奇异']):
    for model in models:
        # We only take the two corresponding features
        X = x.loc[:, pair]
        X = np.array(X)
        y = df1['type']
        y = np.array(y)

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
#         round(a,2）

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std
        df = DataFrame(data=X, columns=pair)
        df.fillna(0.0, inplace=True)
        df[pair[0]].apply(lambda a: round(a, 2))
        df[pair[1]].apply(lambda a: round(a, 2))
#         print(df.describe())
        X = np.array(df)

        # Train
        clf = clone(model)
        clf = model.fit(X, y)

        scores = clf.score(X, y)
        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(
            ".")[-1][:-2][:-len("Classifier")]

        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(
                len(model.estimators_))
        print(model_details + " with features", pair,
              "has a score of", scores)

        plt.subplot(3, 4, plot_idx)
        if plot_idx <= len(models):
            # Add a title at the top of each column
            plt.title(model_title)

        # Now plot the decision boundary using a fine mesh as input to a
        # filled contour plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        # Plot either a single DecisionTreeClassifier or alpha blend the
        # decision surfaces of the ensemble of classifiers
        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            # Choose alpha blend level with respect to the number
            # of estimators
            # that are in use (noting that AdaBoost can use fewer estimators
            # than its maximum if it achieves a good enough fit early on)
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a
        # black outline
        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, plot_step_coarser),
            np.arange(y_min, y_max, plot_step_coarser))
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
                                         yy_coarser.ravel()]
                                         ).reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15,
                                c=Z_points_coarser, cmap=cmap,
                                edgecolors="none")

        # Plot the training points, these are clustered together and have a
        # black outline
        plt.scatter(X[:, 0], X[:, 1], c=y,
                    cmap=ListedColormap(['g', 'y', 'b']),
                    edgecolor='k', s=20)
        plot_idx += 1  # move on to the next plot in sequence

plt.suptitle("Classifiers on feature subsets of the QianZhan")
plt.axis("tight")

plt.show()
