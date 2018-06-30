#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise
from sklearn import datasets, model_selection, svm, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

import seaborn as sns
import pandas as pd

sns.set(style="ticks")

def pred_and_plot(x_train,label_train,x_test,label_test,C,gamma,area):
    area_rows = 2
    area_cols = 3
    svc = svm.SVC(kernel='rbf',C=C,gamma=gamma)
    svc.fit(x_train, label_train)
    
    plt.subplot(area_rows,area_cols,area)
    #テストデータ 
    plt.scatter(x_train[:,0],x_train[:,1],c = label_train,cmap=plt.cm.Spectral,alpha=0.8)

    #分類境界
    plt.title(u"train $\gamma:%.2f$, $C:%.1f$"%(gamma,C))
    XX, YY = np.mgrid[-1.5:1.5:200j, -1.5:1.5:200j]
    Z = svc.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    plt.axis('equal')
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])
    #サポートベクトル
#    plt.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=80,lw=0.5, facecolors='none', zorder=10, edgecolors='k',label='support vector',alpha=0.3)
    plt.legend()
    
    # 分類境界
#    plt.pcolormesh(XX, YY, Z,alpha=0.1)
    CS = plt.contour(XX, YY, Z, colors=['red'],linestyles=['-'], levels=[ 0])
    
    #予測
    pre = svc.predict(x_test)
    plt.subplot(area_rows,area_cols,area+area_cols)
    plt.title(u"predict")
    plt.axis('equal')
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])
    
    
    #分類境界
#    plt.pcolormesh(XX, YY, Z,alpha=0.1)
    CS = plt.contour(XX, YY, Z, colors=['red'],linestyles=['-'], levels=[ 0])
    fmt = {-0.5:"",0:"boundary",0.5:""}
    plt.clabel(CS, CS.levels[::1], inline=True, fmt=fmt, fontsize=10)
    
    #予測
    plt.scatter(x_test[:,0],x_test[:,1],c = pre,cmap=plt.cm.Spectral,alpha=0.8)
    


dataN = 200
x_train,label_train = datasets.make_circles(dataN,factor = 0.4,noise = 0.3)
x_test,label_test = datasets.make_circles(dataN,factor = 0.4,noise = 0.0)


#params = [{'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}]

C = 10000
gamma = 5
pred_and_plot(x_train,label_train,x_test,label_test,C,gamma,1)

C = 1
gamma = 5
pred_and_plot(x_train,label_train,x_test,label_test,C,gamma,2)

C = 1
gamma = 1000
pred_and_plot(x_train,label_train,x_test,label_test,C,gamma,3)

#gridSearch = GridSearchCV(svm.SVC(kernel='rbf'), params, cv=5, scoring='f1_weighted')
#gridSearch.fit(x_train, label_train)
#
#result = pd.DataFrame(gridSearch.grid_scores_)

plt.tight_layout()

plt.savefig("svc_paramTuning.png",dpi=200)