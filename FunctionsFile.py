import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import f


# FUNCTIONS TO TEST THE EXPONENTIALITY OF THE DISTRIBUTION OF TRANSFORMED ELEVATIONS
# PART 1 OF FISHERNESS GOODNESS OF FIT TEST

# vector = np.array([0,0,1]).reshape((3,1))
def Rx(theta, vector):
    # angles should be in radians
    # make sure we have an np array
    if type(vector) is np.ndarray:
        pass  # this works for (N,M) and (N,) shapes
    else:
        vector = np.array(vector)
    rotmat = np.array([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)],
                       [0, np.sin(theta), np.cos(theta)]])
    # result = rotmat @ vector
    result = np.matmul(rotmat, vector)
    return result


def Ry(theta, vector):

    if type(vector) is np.ndarray:
        pass
    else:
        vector = np.array(vector)
    rotmat = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
    result = np.matmul(rotmat, vector)
    return result


def Rz(theta, vector):

    if type(vector) is np.ndarray:
        pass
    else:
        vector = np.array(vector)
    rotmat = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])
    result = np.matmul(rotmat, vector)
    return result


def Cart2Polar(cartesian):

    if type(cartesian) is np.ndarray:
        pass
    else:
        cartesian = np.array(cartesian)
    lengths = np.linalg.norm(cartesian, axis=0)
    if cartesian.shape == (len(cartesian),) or cartesian.shape == (1, len(cartesian)):  # this is a single vector
        cartesian = cartesian.reshape((len(cartesian),))
        polar = np.zeros(cartesian.shape[0])
        polar[1] = np.arctan2(cartesian[1], cartesian[0])  # range in [-pi, pi]
        polar[0] = np.arccos(cartesian[2] / lengths)
        polar[polar < 0] = 2 * np.pi + polar[polar < 0]
    else:
        polar = np.zeros((2, cartesian.shape[1]))
        for i in range(cartesian.shape[1]):
            polar[1,i] = np.arctan2(cartesian[1, i], cartesian[0, i])  # range in [-pi, pi]
            polar[0,i] = np.arccos(cartesian[2, i] / lengths[i])
        polar[1, :][polar[1, :]<0] = 2 * np.pi + polar[1, :][polar[1, :]<0]
    return polar


def Polar2Cart(polar):
    # assume that the polar angles are in radians and in the range [0,pi] [0,2pi]
    # TBD set a flag for whether the range of angles is in the right format
    if isinstance(polar, pd.DataFrame):
        # assume that the order of elements in data frame is (azimuth, elevation), (phi, theta)
        xval = np.array(np.sin(polar.iloc[:, 1]) * np.cos(polar.iloc[:, 0]))
        yval = np.array(np.sin(polar.iloc[:, 1]) * np.sin(polar.iloc[:, 0]))
        zval = np.array(np.cos(polar.iloc[:, 1]))
        xval = xval.reshape((1, xval.shape[0]))  # make into a 1xN matrix for concatenation
        yval = yval.reshape((1, yval.shape[0]))
        zval = zval.reshape((1, zval.shape[0]))
        cartesian = np.concatenate((xval, yval, zval), axis=0)
    else:
        # assume that the angles are in (theta, phi) format
        polar = np.array(polar)
        if polar.shape == (len(polar),) or polar.shape == (1, len(polar)):  # this is a single vector
            polar = polar.reshape((len(polar),))
            xval = np.sin(polar[0]) * np.cos(polar[1])
            yval = np.sin(polar[0]) * np.sin(polar[1])
            zval = np.cos(polar[0])
            cartesian = np.array([xval, yval, zval])
        else:
            if polar.shape[0] > 2:
                polar = polar.T  # now we have a 2xN matrix with column vectors with (phi, theta).T
            xval = np.array(np.sin(polar[0, :]) * np.cos(polar[1, :]))
            yval = np.array(np.sin(polar[0, :]) * np.sin(polar[1, :]))
            zval = np.array(np.cos(polar[0, :]))
            xval = xval.reshape((1, xval.shape[0]))  # make into a 1xN matrix for concatenation
            yval = yval.reshape((1, yval.shape[0]))
            zval = zval.reshape((1, zval.shape[0]))
            cartesian = np.concatenate((xval, yval, zval),axis=0)
    return cartesian


def Test3Rot(theta, phi, vectors, radians=True):
    # radians argument is only relevant if the input is a matrix with column vectors in polar coordinates
    if type(vectors) is np.ndarray:
        pass
    else:
        vectors = np.array(vectors)
    # check if vectors are in polar coordinates or cartesian
    if vectors.shape == (len(vectors),) or vectors.shape == (1,vectors.shape[1]):  # this is a single vector
        vectors = vectors.reshape(-1)  # make it into a (N,) array, not 1xN matrix
        if len(vectors) == 2:  # this is a polar coordinate vector
            # first, convert to radians
            # if sum(np.logical_and(abs(vectors) >= 0, abs(vectors) <= 2*np.pi)) == len(a)
            if not radians:
                vectors = vectors / 180 * np.pi  # convert to radians
                vectors[vectors<0] = 2*np.pi + vectors[vectors<0]
            else:  # this wont do anything if the degrees are all positive already
                vectors[vectors < 0] = 2 * np.pi + vectors[vectors < 0]
            # convert to cartesian
            ele = vectors[0]
            azi = vectors[1]
            cartesian = Polar2Cart(vectors)
            zeroedcart = Ry(-ele, Rz(-azi, cartesian))
            movedcart = Rz(phi, Ry(theta, zeroedcart))

        else:  # this is a cartesian vector
            polar = Cart2Polar(vectors)
            ele = polar[0]
            azi = polar[1]
            zeroedcart = Ry(-ele, Rz(-azi, vectors))
            movedcart = Rz(phi, Ry(theta, zeroedcart))
    elif vectors.shape[0] == 2:  # this is a matrix with column vectors in polar coordinates
        if not radians:
            vectors = vectors / 180 * np.pi  # convert to radians
            vectors[vectors < 0] = 2 * np.pi + vectors[vectors < 0]
        else:  # this wont do anything if the degrees are all positive already
            vectors[vectors < 0] = 2 * np.pi + vectors[vectors < 0]
        cartesian = Polar2Cart(vectors)
        centre_of_mass = np.mean(cartesian, axis=1)
        mean_res_length = np.linalg.norm(centre_of_mass)
        mean_res_dirn = centre_of_mass / mean_res_length
        mean_polar = Cart2Polar(mean_res_dirn)
        ele = mean_polar[0]
        azi = mean_polar[1]
        # now that we know the polar coordinates of the mean direction, can move vectors to a new location
        zeroedcart = Ry(-ele, Rz(-azi, cartesian))
        movedcart = Rz(phi, Ry(theta, zeroedcart))

    else:  # matrix with column vectors in cartesian coordinates
        centre_of_mass = np.mean(vectors, axis=1)
        mean_res_length = np.linalg.norm(centre_of_mass)
        mean_res_dirn = centre_of_mass / mean_res_length
        mean_polar = Cart2Polar(mean_res_dirn)
        ele = mean_polar[0]
        azi = mean_polar[1]
        # now that we know the polar coordinates of the mean direction, can move vectors to a new location
        zeroedcart = Ry(-ele, Rz(-azi, vectors))
        movedcart = Rz(phi, Ry(theta, zeroedcart))
    return movedcart


def Xi(elearray: object) -> object:
    elearray = np.array(elearray)
    result = 1 - np.cos(elearray)
    return result


def kmle(thetas):
    # the thetas have to be transformed before input to this function, sample mean direction is the pole theta' = 0
    if type(thetas) is np.ndarray:
        pass
    else:
        thetas = np.array(thetas)
    n = len(thetas)
    k = (n-1) / sum(1 - np.cos(thetas))
    return k


def F(k, elevation):
    # elevation is transformed with Xi before being input to this function
    if type(elevation) is np.ndarray:
        pass
    else:
        elevation = np.array(elevation)
    elevation = np.sort(elevation)
    result = 1 - np.exp(-k*elevation)
    return result


def Dnplus(FX):
    # use F for the exponentiality test and F2 for normality test as the 'function' argument
    n = len(FX)
    array1 = np.zeros(len(FX))
    for i in range(len(FX)):
        array1[i] = i/n - FX[i]
    max1 = np.max(array1)
    return max1


def Dnminus(FX):
    # use F for the exponentiality test and F2 for normality test as the 'function' argument
    n = len(FX)
    array2 = np.zeros(len(FX))
    for i in range(len(FX)):
        array2[i] = FX[i] - (i - 1)/n
    max2 = np.max(array2)
    return max2


def Dn(max1, max2):
    Dnmax = np.max([max1, max2])
    return Dnmax


def Me(Dnmax, n):
    # this is a function for Kolmogorov Smirnov statistic to test for exponentiality of elevation
    result = (Dnmax - 0.2/n) * (np.sqrt(n) + 0.26 + 0.5/np.sqrt(n))
    return result


def Mn(Dnmax, n):
    # this is a function for Kolmogorov Smirnov statistic to test for exponentiality of elevation
    result = Dnmax * (np.sqrt(n) - 0.01 + 0.85/np.sqrt(n))
    return result


# FUNCTIONS TO TEST THE UNIFORMITY OF AZIMUTHS
# PART 2 OF THE FISHERNESS GOODNESS OF FIT TEST
def F2(azimuth):
    # angles have to be in radians
    if type(azimuth) is np.ndarray:
        pass
    else:
        azimuth = np.array(azimuth)
    azimuth = np.sort(azimuth)
    result = azimuth/(2*np.pi)
    return result


def Vn(Dplus, Dminus):
    result = Dplus+Dminus
    return result


def MV(V, n):
    result = V*(np.sqrt(n) - 0.467 + 1.623/np.sqrt(n))
    return result


# TEST FOR EQUALITY OF MEAN DIRECTIONS WITH EQUAL K, FISHER BOOK 7.2.3
def FisherMeanTestSameK(samples):
    # samples is a tuple of cartesian coordinates
    sample1 = np.array(samples[0])
    sample2 = np.array(samples[1])

    if sample1.shape == (len(sample1),) or sample1.shape == (1,sample1.shape[1]):  # this is a single vector
        sample1 = sample1.reshape(-1)
        n1 = len(sample1)
    else:
        n1 = sample1.shape[1]
    if sample2.shape == (len(sample2),) or sample2.shape == (1,sample2.shape[1]):  # this is a single vector
        sample2 = sample2.reshape(-1)
        n2 = len(sample2)
    else:
        n2 = sample2.shape[1]

    N = n1+n2

    resultant1 = np.sum(sample1,axis=1)
    resultant2 = np.sum(sample2,axis=1)
    resultant = resultant1 + resultant2

    res_len = np.linalg.norm(resultant)
    res_len1 = np.linalg.norm(resultant1)
    res_len2 = np.linalg.norm(resultant2)
    Z = (res_len1 + res_len2)/N
    mean_res_length = res_len / N
    dof1 = 2
    dof2 = 2*N - 4
    if n1 == n2:
        if mean_res_length <0.75:
            # in this case find z0 in the A20 appendix in Fisher book and reject H0 that the means are all equal if
            # Z exceeds z0
            return Z, mean_res_length
        else:
            # this gives an upper percent point, with probability of exceeding it of alpha
            f_upp_crit = f.ppf(0.95, dof1, dof2)
            # lower percent point is obtained with f.ppf(0.05,2,8) or 1/f.ppf(0.95,8,2)
            z0 = (mean_res_length + f_upp_crit/(N-2)) / (1 + f_upp_crit/(N-2))
            return Z, z0, 1
    else:
        pass





# FUNCTIONS TO TEST THE NORMALITY OF LINEAR DATA
def AndersonDarling(sample):
    # this test can be used as a goodness of fit test with many distributions, this one concerns normal distribution
    n = len(sample)
    order_statistic = np.sort(sample)
    sample_mean = np.mean(order_statistic)
    sample_var = 1/(n - 1) * ((order_statistic - sample_mean) @ (order_statistic - sample_mean))
    sstd = np.sqrt(sample_var)
    std_ord_stat = (order_statistic - sample_mean) / sstd
    summed = 0
    for i in range(1, n+1):
        summed += (2*i - 1) * (np.log(norm.cdf(std_ord_stat[i-1])) + np.log(1 - norm.cdf(std_ord_stat[n-i])))
    A2 = -n - 1/n*summed
    # this should be used when both true mean and var are unknown
    A2corrected = A2 * (1 + 4/n - 25/n**2)  # from Stephens 1974
    # A2corrected = A2 * (1 + 0.75 / n + 2.25 / n ** 2)  # from D'Agostino 1986 in Table 4.7 on p. 123 and p. 372–373
    # this test is not so sensitive to uniform as to say exponential distribution, which makes sense intuitively
    return A2corrected



# a = Rx(np.pi/2, vector) 0.6628932090204712
# print (a)

# FUNCTIONS FOR LINEAR STATISTICS DATA TESTS
def Ttest2Samples(sample1, sample2, alpha=0.05):
    # both samples should be 1d arrays or matrices
    if type(sample1) is np.ndarray and type(sample2) is np.ndarray:
        pass
    else:
        sample1 = np.array(sample1)
        sample2 = np.array(sample2)
    sample1 = sample1.reshape(-1)
    sample2 = sample2.reshape(-1)

    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    n1 = len(sample1)
    n2 = len(sample2)
    svar1 = (sum(sample1**2) - sum(sample1)**2/n1)/(n1-1)
    svar2 = (sum(sample2 ** 2) - sum(sample2) ** 2 / n2) / (n2 - 1)
    sstd1 = np.sqrt(svar1)
    sstd2 = np.sqrt(svar2)
    SE1 = sstd1/np.sqrt(n1)
    SE2 = sstd2 / np.sqrt(n2)
    #sp**2 is a pooled variance estimate
    if n1 == n2:
        sp = np.sqrt(SE1**2 + SE2**2)
    else:
        sp = np.sqrt((1/n1 + 1/n2) * ((n1-1) * svar1 + (n2-1) * svar2)/(n1+n2-2))
    dof = n1 + n2 - 2
    tstatistic = (mean1 - mean2)/sp
    tval = t.ppf(alpha/2, dof)  # this is for two tailed test, TDB include one tailed test
    return np.array([tstatistic, tval, alpha])


def FindAngle(vector1, vector2):
    cos = np.dot(vector1, vector2) / np.linalg.norm(vector1) * np.linalg.norm(vector2)
    return np.arccos(cos) * 180 / np.pi
# sample1 = (sigma1 * np.random.randn(1, 22) + mu1).reshape(-1)
# sample2 = np.random.uniform(0, 1, 22)
# sample3 = np.random.exponential(1, 22)
