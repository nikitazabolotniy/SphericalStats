from typing import Union

import numpy as np
import pandas as pd

#g = pd.read_csv('/Users/nikitazabolotniy/Calliphora_F_H1.txt', header=None)
from numpy.core._multiarray_umath import ndarray

g1 = pd.read_csv('/Users/nikitazabolotniy//Hermetia_F_H1.txt', header=None)
g1YJ = pd.read_csv('/Users/nikitazabolotniy/YJHermetia_F_H1.txt', header=None)
g2 = pd.read_csv('/Users/nikitazabolotniy/Hermetia_M_H1.txt', header=None)
g2YJ = pd.read_csv('/Users/nikitazabolotniy/YJHermetia_M_H1.txt', header=None)
g1 = pd.concat([g1, g1YJ], axis=0,join='outer',ignore_index=True)
g2 = pd.concat([g2, g2YJ], axis=0,join='outer',ignore_index=True)
g = pd.concat([g1,g2],axis=0, join='outer',ignore_index=False, keys=['F','M'])
# imported as data frame; g1['157.2482'] gives the first column g1.iloc[0] gives the first row
g = g.iloc[:,[0,1,3,4]]

g[0][g[0]>0] = 360 - g[0][g[0]>0]
g[0][g[0]<0] = g[0][g[0]<0].abs()  # this works for column extraction, not with indexing in strings

g[1][g[1]>0] = 90 - g[1][g[1]>0]  # do this first because here we end up with positive values
# now operate on negative values and turn them positive
g[1][g[1]<0] = g[1][g[1]<0].abs() + 90
# now elevation is [0,pi] azimuth [0,2pi]

# convert into cartesian coordinates

# convert into radians
g = g/180*np.pi  # format azimuth elevation (phi, theta)
g['rotx'] = np.sin(g.iloc[:,1]) * np.cos(g.iloc[:,0])
g['roty'] = np.sin(g.iloc[:,1]) * np.sin(g.iloc[:,0])
g['rotz'] = np.cos(g.iloc[:,1])
g['RotAxes'] = g[['rotx', 'roty', 'rotz']].apply(list, axis=1)  # rows are iterable so can apply list to them

# g['RotAxes'] = g['rotx'].map(str) + ',' + g['roty'].map(str) + ',' + g['rotz'].map(str)
# g['RotAxes'] = [[float(y) for y in x.split(",")] for x in g['RotAxes']]  # this will give a lists of lists
# # where each individual list would form a row of the column
# # g = g.drop(['x','y','z'],axis=1)

cartesian = np.array(g['RotAxes'].tolist())  # converted a list to an ndarray
cartesianF = np.array(g.loc['F']['RotAxes'].tolist())
cartesianM = np.array(g.loc['M']['RotAxes'].tolist())

resultant = np.sum(cartesian,axis=0)
resultantF = np.sum(cartesianF,axis=0)
resultantM = np.sum(cartesianM,axis=0)

res_len = np.linalg.norm(resultant)
res_lenF = np.linalg.norm(resultantF)
res_lenM = np.linalg.norm(resultantM)

centre_of_mass = np.mean(cartesian,axis=0)
centre_of_massF = np.mean(cartesianF,axis=0)
centre_of_massM = np.mean(cartesianM,axis=0)

mean_res_length = np.linalg.norm(centre_of_mass)
mean_res_lengthF = np.linalg.norm(centre_of_massF)
mean_res_lengthM = np.linalg.norm(centre_of_massM)

n = g.shape[0]
n1 = g.loc['F'].shape[0]
n2 = g.loc['M'].shape[0]
p = 3
q=2
# TEST FOR EQUALITY OF MEAN DIRECTIONS for large k (10.5.1) F{p-1,(p-1)*(n-2)}
# assume equality of concentration parameters 
High_k_Ftest = ((res_lenF+res_lenM - res_len)/(p-1))/((n - res_lenF - res_lenM)/((n-2)*(p-1)))
# at 5% confidence, critical values of Ftest is 3.232, statistic above gives 0.31, so we cant reject H0,
# hence we accept H0 and conclude that the means are equal for males and females, k are 753 for F and 250 for M

# TEST FOR EQUALITY OF CONCENTRATION PARAMETERS mean directions mu1 and m0 unknown (10.5.2 in the book)
F = ((n1 - res_lenF)/((n1-1)*(p-1)))/((n2 - res_lenM)/((n2-1)*(p-1)))
# F test for 14,6 DoF gives 5.12 upper and 0.35 lower critical values, F = 0.2985 so we reject H0
# stating that the concentration parameters are equal with 98% confidence
# but also want to test the Fisherness of the combined sample

# ANOVA BASED ON THE EMBEDDING APPROACH (assume ks are equal and large)
A = ((n1*mean_res_lengthF**2 + n2*mean_res_lengthM**2 - mean_res_length**2)/((q-1)*(p-1)))/ \
    ((n - n1*mean_res_lengthF**2 + n2*mean_res_lengthM**2)/((n-q)*(p-1)))
# this test rejects H0, maybe due to a large difference in k

# for p=3 and R>=0.9
# mardia 10.3.7, for large k
k = (p-1)/(2*(1-mean_res_length))  # 190.8647162997246
# any p and R?
k1 = mean_res_length*(p-mean_res_length**2)/(1-mean_res_length**2)  # 191.3581637180914, formula from wiki
# mardia 10.3.25 gives an approximately unbiased estimator
# for large k and n
kunb = ((n-1)*(p-1) - 2)/(2*n*(1 - mean_res_length))
# k = 100 via A^{-1}(R)
# for p=3 and R>=0.9
kF = (p-1)/(2*(1-mean_res_lengthF))  # 190.8647162997246
k1F = mean_res_length*(p-mean_res_lengthF**2)/(1-mean_res_lengthF**2)  # 191.3581637180914, formula from wiki
# k = 100 via A^{-1}(R)
# for p=3 and R>=0.9
kM = (p-1)/(2*(1-mean_res_lengthM))  # 190.8647162997246
k1M = mean_res_length*(p-mean_res_lengthM**2)/(1-mean_res_lengthM**2)  # 191.3581637180914, formula from wiki
# k = 100 via A^{-1}(R)

# after 10.3.25 there's an approximately unbiased estimator unless both n and k are small
kapunb = (1 - 1/n1)**2 * (p-1) / (2*(1 - mean_res_lengthF))

# test for mean direction
mean_direction = np.array([0,0,-1])
# dot prod, equivalently use a@b for np arrays
n = g.shape[0]  # number of samples
C = sum(centre_of_mass[i]*mean_direction[i] for i in range(len(centre_of_mass)))
w = 2*n*k*(mean_res_length - C)  # 36.94780842802157  ONLY WHEN k IS KNOWN
# we reject the H0 with 0.002 significance level, so mean is not [0,0,-1]
# asymptotic (large sample or large k) distribution of w is chi_squared(p)
