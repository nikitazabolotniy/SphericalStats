import numpy as np
from Rot3D import Rz
from Rot3D import Ry
from scipy.stats import beta


def FisherDistribution(k, n, ele=0, azi=0):
    lam = np.exp(-2*k)
    cartesian_sample = np.zeros((n, 3))
    for i in range(n):
        r = np.random.uniform(0,1,2)
        r1 = r[0]
        r2 = r[1]
        theta = 2*np.arcsin(np.sqrt(-np.log(r1*(1-lam) + lam)/(2*k)))
        if theta < 0: theta = abs(theta) + 90
        phi = 2*np.pi*r2
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        cartesian_sample[i,:] = np.array([x,y,z])
    # if we want to rotate the data then apply this
    if ele == 0 and azi == 0:
        pass
    else:
        cartesian_sample = cartesian_sample.T
        transformed = Rz(azi, Ry(ele, cartesian_sample))
        cartesian_sample = transformed.T
    return cartesian_sample

    # now rotate to a desired position
# a = FisherDistribution(3, 25)
# print(a)
#
# print('mean is {}'.format(np.mean(a,axis=0)))

def FisherDistribution2(k, n, m, ele=0, azi=0):
    # m is the number of dimensions
    b = (-2 * k + np.sqrt(4 * k ** 2 + (m - 1) ** 2)) / (m - 1)
    x0 = (1 - b) / (1 + b)
    c = k * x0 + (m - 1) * np.log(1 - x0 ** 2)
    numvec = 0
    cartesian_sample = np.zeros((n, m))
    while numvec < n:
        Z = beta.rvs((m - 1) / 2, (m - 1) / 2, size=1)
        U = np.random.uniform(0, 1, 1)
        W = (1 - (1 + b) * Z) / (1 - (1 - b) * Z)
        if k*W + (m - 1) * np.log(1 - x0 * W) - c < np.log(U):
            continue
        else:
            theta = 2 * np.pi * np.random.uniform(0, 1, 1)[0]
            V = np.array([np.cos(theta), np.sin(theta)])  # 2d vector
            X = np.concatenate((np.sqrt(1 - W**2) * V, W))
            cartesian_sample[numvec, :] = X
            numvec += 1  # count the free rows in cartesian, when fully filled - quit

    if ele == 0 and azi == 0:
        pass
    else:
        cartesian_sample = cartesian_sample.T
        transformed = Rz(azi, Ry(ele, cartesian_sample))
        cartesian_sample = transformed.T

    return cartesian_sample

# a = FisherDistribution2(3, 25, 3)
# print(a)
#
# print('mean is {}'.format(np.mean(a,axis=0)))
