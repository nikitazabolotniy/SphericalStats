import numpy as np
import pandas as pd
import Rot3D
from scipy.stats import norm

#THIS IS A GOODNESS OF FIT TEST FOR FISHERNESS
g1 = pd.read_csv('/Users/nikitazabolotniy/Hermetia_F_H1.txt', header=None)
g1YJ = pd.read_csv('/Users/nikitazabolotniy/YJHermetia_F_H1.txt', header=None)
g2 = pd.read_csv('/Users/nikitazabolotniy/Hermetia_M_H1.txt', header=None)
g2YJ = pd.read_csv('/Users/nikitazabolotniy/YJHermetia_M_H1.txt', header=None)
g1 = pd.concat([g1, g1YJ], axis=0,join='outer',ignore_index=True)
g2 = pd.concat([g2, g2YJ], axis=0,join='outer',ignore_index=True)
g = pd.concat([g1,g2],axis=0, join='outer',ignore_index=False, keys=['F','M'])
# imported as data frame; g1['157.2482'] gives the first column g1.iloc[0] gives the first row
g = g.iloc[:,[0,1,3,4]]

g[0][g[0]>0] = 360 - g[0][g[0]>0]  # format is azimuth, elevation
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

# represent vectors as rows of a matrix, have to use their transposes for rotation matrices
cartesian = np.array(g['RotAxes'].tolist())  # converted a list to an ndarray
cartesianF = np.array(g.loc['F']['RotAxes'].tolist())
cartesianM = np.array(g.loc['M']['RotAxes'].tolist())

resultant = np.sum(cartesian,axis=0)
resultantF = np.sum(cartesianF,axis=0)
resultantM = np.sum(cartesianM,axis=0)

res_len = np.linalg.norm(resultant)
res_lenF = np.linalg.norm(resultantF)
res_lenM = np.linalg.norm(resultantM)

Npole = np.array([0,0,1])
centre_of_mass = np.mean(cartesian,axis=0)
centre_of_massF = np.mean(cartesianF,axis=0)
centre_of_massM = np.mean(cartesianM,axis=0)

mean_res_length = np.linalg.norm(centre_of_mass)
mean_res_lengthF = np.linalg.norm(centre_of_massF)
mean_res_lengthM = np.linalg.norm(centre_of_massM)

azi = np.arctan2(centre_of_mass[1], centre_of_mass[0])  # range in [-pi, pi]
if azi < 0 :
    azi = 2*np.pi + azi
ele = np.arccos(centre_of_mass[2] / mean_res_length)

cartesian = cartesian.T
movedmean = Rot3D.Ry(-ele, Rot3D.Rz(-azi, centre_of_mass))
movedcart = Rot3D.Ry(-ele, Rot3D.Rz(-azi, cartesian))
movedcart = movedcart.T  # convert back to (3,N) array
movedlen = np.linalg.norm(movedcart,axis=1)  # find lengths of moved vectors
# check how much different is the moved mean from computed moved mean, if very similar then everything is correct
# difference = movedmean - np.mean(movedcart,axis=0)

movedazi = np.zeros(movedcart.shape[0])
movedele = np.zeros(movedcart.shape[0])

for i in range(len(movedazi)):
    movedazi[i] = np.arctan2(movedcart[i, 1], movedcart[i, 0])  # range in [-pi, pi]
    if azi < 0 :
        azi = 2*np.pi + azi
    movedele[i] = np.arccos(movedcart[i, 2] / movedlen[i])

Xi = Rot3D.Xi(movedele)
k = Rot3D.kmle(movedele)  # mle for concentration parameter k
FX = Rot3D.F(k, Xi)
# compute the order statistic of  Xi before inputting them to the F function, this is done in the Dn function
n = movedele.shape[0]
Dplus = Rot3D.Dnplus(FX)
Dminus = Rot3D.Dnminus(FX)
Dn = Rot3D.Dn(Dplus, Dminus)
statistic = Rot3D.Me(Dn, n)
# 0.6628932090204712 so we accept the hypothesis of exponential distribution
print(statistic)

FX2 = Rot3D.F2(movedazi)
# movedazi[movedazi<0] = movedazi[movedazi<0] + 2*np.pi  test statistic doesnt change if we convert to positive radians
D2plus = Rot3D.Dnplus(FX2)
D2minus = Rot3D.Dnminus(FX2)
Vn = D2plus + D2minus
statistic2 = Rot3D.MV(Vn, n)
# 0.7407577735624274 so we accept the hypothesis of uniform distribution

movedmean3 = Rot3D.Ry(np.pi/2 - ele, Rot3D.Rz(-azi, centre_of_mass))
movedcart3 = Rot3D.Ry(np.pi/2 - ele, Rot3D.Rz(-azi, cartesian))
movedcart3 = movedcart3.T  # convert back to (3,N) array
movedlen3 = np.linalg.norm(movedcart3,axis=1)
movedazi3 = np.zeros(movedcart3.shape[0])
movedele3 = np.zeros(movedcart3.shape[0])

for i in range(len(movedazi)):
    movedazi3[i] = np.arctan2(movedcart3[i, 1], movedcart3[i, 0])  # range in [-pi, pi]
    movedele3[i] = np.arccos(movedcart3[i, 2] / movedlen3[i])

Xi3 = np.multiply(movedazi3, np.sqrt(np.sin(movedele3)))#phi*sqrt sin
variance = (Xi3 @ Xi3)/n
normalisedXi3 = Xi3 / variance
FX3 = norm.cdf(normalisedXi3)
FX3 = np.sort(FX3)
Dplus3 = Rot3D.Dnplus(FX3)
Dminus3 = Rot3D.Dnminus(FX3)
Dn3 = Rot3D.Dn(Dplus3, Dminus3)
statistic3 = Rot3D.Mn(Dn3, n)
# 2.4163260985892676, so we can accept normality of the product with 95% confidence but not with 99%

# TEST THE STATISTIC DEFINED IN THE PAPER BASED ON EIGENVALUES


n = g.shape[0]
n1 = g.loc['F'].shape[0]
n2 = g.loc['M'].shape[0]
p = 3
q=2

# test data
a = np.array([[0.388, 0.171, 0.272, 0.123, 0.182, 0.291, 0.225, 0.518, 0.449],
              [0.117, -0.321, -0.204, -0.062, 0.003, -0.029, -0.272, 0.022, -0.433],
              [-0.914, -0.932, -0.940, -0.991, -0.983, -0.956, -0.935,-0.855, -0.782]])
a = a.T

