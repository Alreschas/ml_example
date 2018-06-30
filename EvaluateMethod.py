# -*- coding: utf-8 -*-

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

y_true = [0,0,0, 0,0,0,1,1,1,1]
y_pred = [0,1,1, 1,1,1,1,1,1,1]
mat = confusion_matrix(y_true, y_pred)

print("confmat")
print(mat)

print("############")
print("report")
print(classification_report(y_true, y_pred))

print("accuracy:",accuracy_score(y_true, y_pred, normalize=True))

f1_macro = f1_score(y_true, y_pred, average='macro')
f1_micro = f1_score(y_true, y_pred, average='micro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
print("f1 macro:%.2f,micro:%.2f,weighted:%.2f"%(f1_macro,f1_micro,f1_weighted))

#正解データ数
sup0 = y_true.count(0)
sup1 = y_true.count(1)

#recall(適合率) 正解と判断した中で、本当に正解だった割合
rec0 = 0
rec1 = 0
pre0 = 0
pre1 = 0
f10 = 0
f11 = 0
if(mat[0,0]+mat[0,1]!= 0):
    rec0 = mat[0,0]/(mat[0,0]+mat[0,1])
if(mat[1,0]+mat[1,1]!= 0):
    rec1 = mat[1,1]/(mat[1,0]+mat[1,1])

#precision(正確性) 正解データ 全部の中で、正解と判断した割合
if(mat[0,0]+mat[1,0]!= 0):
    pre0 = mat[0,0]/(mat[0,0]+mat[1,0])
if(mat[0,1]+mat[1,1]!= 0):
    pre1 = mat[1,1]/(mat[0,1]+mat[1,1])


#F値
if(rec0+pre0 != 0):
    f10 = 2*rec0*pre0/(rec0+pre0)
if(rec1+pre1 != 0):
    f11 = 2*rec1*pre1/(rec1+pre1)

f1_macro = (f10+f11)/2

#F1microは、２クラスではAccuracyと同じ
f1_micro = (mat[0,0]+mat[1,1])/(mat[0,0]+mat[0,1]+mat[1,0]+mat[1,1])

f1_weighted = 0
if(sup0+sup1 != 0):
    f1_weighted = (f10 * sup0 + f11 * sup1)/(sup0+sup1)

print("############")
print('report')
print('\t   precision','  recall','  f1-score','  support')
print()
print("\t%d\t%.2f\t%.2f\t%.2f\t\t%d"%(0,pre0,rec0,f10,sup0))
print("\t%d\t%.2f\t%.2f\t%.2f\t\t%d"%(1,pre1,rec1,f11,sup1))
print("f1 macro:%.2f,micro:%.2f,weighted:%.2f"%(f1_macro,f1_micro,f1_weighted))