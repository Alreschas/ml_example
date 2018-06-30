#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise
from sklearn import datasets, model_selection, svm, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
import pandas as pd



#sns.set(style="ticks")

def project_data(data):
    x=data[:,0]
    y=data[:,1]
    z=x**2+y**2
    return x,y,z


dataN = 200
x_train,label_train = datasets.make_circles(dataN,factor = 0.3,noise = 0.0)
x_test,label_test = datasets.make_circles(dataN,factor = 0.3,noise = 0.0)

#データ を３Dにマッピング
x,y,z=project_data(x_train)

fig = plt.figure()
#データ プロット
ax = fig.add_subplot(221)
ax.set_title("training data")
ax.axis('equal')
ax.axis([-1, 1, -1, 1])
ax.scatter(x_train[:,0],x_train[:,1],c = label_train,cmap=plt.cm.bwr)



#2Dの場合
"""
ax = fig.add_subplot(122)
ax.axis('equal')
ax.axis([-1, 1, 0.5, 1])

ax.scatter(x[label_train==0],z[label_train==0],c='b')
ax.scatter(x[label_train==1],z[label_train==1],c='r')

#線形SVC
svc = svm.SVC(kernel='linear',C=10000)
svc.fit(np.array([x,z]).T,label_train)

#分類境界をプロット
XX, YY = np.mgrid[-1.:1.:10j, 0:1.:10j]
Z = svc.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)
CS = plt.contour(XX, YY, Z, colors=['red'],linestyles=['-'], levels=[ 0])


#サポートベクトルをプロット
plt.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=90,lw=1.0, facecolors='none', zorder=10, edgecolors='k',label='support vector',alpha=1.0)
plt.legend()
"""
    
#分類境界

#3Dの場合
### 基準
ax = fig.add_subplot(222, projection='3d')
ax.set_title("gaussian")
svc = svm.SVC(kernel='rbf',C=10,gamma=100)
svc.fit(np.array([x,y]).T,label_train)
Z = svc.decision_function(np.c_[x,y])
ax.scatter(x[Z>0],y[Z>0],Z[Z>0],c='r',zorder=1)
ax.scatter(x[Z<0],y[Z<0],Z[Z<0],c='b',zorder=1)

#分類境界
XX, YY = np.mgrid[-1.5:1.5:10j, -1.5:1.5:10j]
Z = np.zeros([XX.shape[0], XX.shape[1]])
Z = Z.reshape(XX.shape)
ax.plot_surface(XX, YY, Z,linewidth=0, antialiased=True, cmap=plt.cm.hot, alpha=0.5,zorder=2)


### 多項式カーネル
ax = fig.add_subplot(223, projection='3d')
ax.set_title("polynomial")
svc = svm.SVC(kernel='poly',C=1,gamma=2)
svc.fit(np.array([x,y]).T,label_train)
Z = svc.decision_function(np.c_[x,y])
ax.scatter(x[Z>0],y[Z>0],Z[Z>0],c='r',zorder=1)
ax.scatter(x[Z<0],y[Z<0],Z[Z<0],c='b',zorder=1)

#分類境界
XX, YY = np.mgrid[-1.5:1.5:10j, -1.5:1.5:10j]
Z = np.zeros([XX.shape[0], XX.shape[1]])
Z = Z.reshape(XX.shape)
ax.plot_surface(XX, YY, Z,linewidth=0, antialiased=True, cmap=plt.cm.hot, alpha=0.5,zorder=2)


### ガンマを増やす
ax = fig.add_subplot(224, projection='3d')
ax.set_title("linear")
svc = svm.SVC(kernel='linear',C=1)
svc.fit(np.array([x,y]).T,label_train)
Z = svc.decision_function(np.c_[x,y])
ax.scatter(x[Z>0],y[Z>0],Z[Z>0],c='r',zorder=1)
ax.scatter(x[Z<0],y[Z<0],Z[Z<0],c='b',zorder=1)

#分類境界
XX, YY = np.mgrid[-1.5:1.5:10j, -1.5:1.5:10j]
Z = np.zeros([XX.shape[0], XX.shape[1]])
Z = Z.reshape(XX.shape)
ax.plot_surface(XX, YY, Z,linewidth=0, antialiased=True, cmap=plt.cm.hot, alpha=0.5,zorder=2)


#Z = svc.decision_function(np.c_[x[label_train==1],y[label_train==1]])
#ax.scatter(x[label_train==1],y[label_train==1],Z,c='r',zorder=2)

plt.savefig("SVC_kernels.png",dpi=200)


plt.tight_layout()
