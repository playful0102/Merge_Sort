import pandas

names = ['Alcohol', 'Flavanoids', 'wineClass']
dataset = pandas.read_csv(r'C:\Users\Anson Fok\Desktop\School\year2 sem2\machine\wine-s.csv', header=0, names=names)
x = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(0)
perm = np.random.permutation(178)
x = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values
trainx = dataset.iloc[perm[0:130], 0:2].values
trainy = dataset.iloc[perm[0:130], 2].values
testx = dataset.iloc[perm[130:178], 0:2].values
testy = dataset.iloc[perm[130:178], 2].values
# print(trainx.shape)
# print(sum(trainy == 1))

from scipy.stats import norm, multivariate_normal

# feature = 0
# label = 1
# plt.hist(trainx[trainy == label, feature], density=True)
# mu = np.mean(trainx[trainy == label, feature])  # mean
# var = np.var(trainx[trainy == label, feature])  # variance
# std = np.sqrt(var)  # standard deviation
# x_axis = np.linspace(mu - 3 * std, mu + 3 * std, 1000)
# plt.plot(x_axis, norm.pdf(x_axis, mu, std), 'r', lw=3)
# plt.title("Winery " + str(label))
# plt.xlabel(names[feature], color='blue')
# plt.ylabel('Density', color='blue')
# plt.show()

# def fit_generative_model(x, y, feature):
#     k = 3  # number of classes
#     mu = np.zeros(k + 1)  # list of means
#     var = np.zeros(k + 1)  # list of variances
#     pi = np.zeros(k + 1)  # list of class weights
#     for label in range(1, k + 1):
#         indices = (y == label)
#         mu[label] = np.mean(x[indices, feature])
#         var[label] = np.var(x[indices, feature])
#         pi[label] = float(sum(indices)) / float(len(y))
#     return mu, var, pi
#
#
# feature = 0
# mu, var, pi = fit_generative_model(trainx, trainy, feature)
# colors = ['r', 'k', 'g']
# for label in range(1, 4):
#     m = mu[label]
#     s = np.sqrt(var[label])
#     x_axis = np.linspace(m - 3 * s, m + 3 * s, 1000)
#     plt.plot(x_axis, norm.pdf(x_axis, m, s), colors[label - 1], label="class " + str(label))
# plt.xlabel(names[feature], fontsize=14, color='red')
# plt.ylabel('Density', fontsize=14, color='red')
# plt.legend()
# plt.show()

from sklearn.neighbors import KNeighborsClassifier

kNN1 = KNeighborsClassifier(n_neighbors=5)
kNN1.fit(trainx, trainy)

x = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values
f1 = 0
f2 = 1
colors = ['r', 'k', 'g']

from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ['darkorange', 'c', 'darkblue']
h = 0.01
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = kNN1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=.8, shading='auto')
for label in range(1, 4):
    plt.plot(x[y == label, f1], x[y == label, f2], marker='o', ls='None', c=colors[label - 1])

# plt.show()

predictions = kNN1.predict(trainx)
from sklearn import metrics
print(metrics.confusion_matrix(trainy, predictions))
