#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise
import seaborn as sns

sns.set(style="ticks")

#多項式カーネル
def poly_kernel(x1,x2,dim):
    ret = np.zeros([x1.shape[0],x2.shape[0]])
    for i in range(x2.shape[0]):
        ret[:,i] = ((1+x1*x2[i])**(dim))[:,0]
    return ret

#ガウスカーネル
def gauss_kernel(x1,x2,beta):
    ret = np.zeros([x1.shape[0],x2.shape[0]])
    for i in range(x2.shape[0]):
        ret[:,i] = np.exp(-beta*(x1-x2[i])**2)[:,0]
    return ret

fig = plt.figure()
fig.suptitle("Gaussian Kernel / Polynomial Kernel", fontsize=16)
x1 = np.atleast_2d(np.linspace(-1,1,100)).T
x2 = np.atleast_2d(np.linspace(-1,1,10)).T

#ガウスカーネルのプロット
ax = fig.add_subplot(321)
ax.title.set_text(r"$\beta=10^0$")
#y = gauss_kernel(x1,x2,10**0)
y = pairwise.rbf_kernel(x1,x2,10**0)
ax.plot(x1,y)

ax = fig.add_subplot(323)
ax.title.set_text(r"$\beta=10^1$")
#y = gauss_kernel(x1,x2,10**1)
y = pairwise.rbf_kernel(x1,x2,10**1)
ax.plot(x1,y)

ax = fig.add_subplot(325)
ax.title.set_text(r"$\beta=10^2$")
#y = gauss_kernel(x1,x2,10**2)
y = pairwise.rbf_kernel(x1,x2,10**2)
ax.plot(x1,y)


#多項式カーネルのプロット
ax = fig.add_subplot(322)
ax.title.set_text(r"$\dim=1$")
#y = poly_kernel(x1,x2,1)
y = pairwise.polynomial_kernel(x1,x2,degree=1)
ax.plot(x1,y)

ax = fig.add_subplot(324)
ax.title.set_text(r"$\dim=5$")
#y = poly_kernel(x1,x2,5)
y = pairwise.polynomial_kernel(x1,x2,degree=5)
ax.plot(x1,y)

ax = fig.add_subplot(326)
ax.title.set_text(r"$\dim=10$")
#y = poly_kernel(x1,x2,10)
y = pairwise.polynomial_kernel(x1,x2,degree=10)
ax.plot(x1,y)

#サブプロットのタイトルが見えるように修正
fig.tight_layout()
fig.subplots_adjust(top=0.88)