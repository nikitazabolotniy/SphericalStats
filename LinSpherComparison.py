import numpy as np
from Rot3D import Rz
from Rot3D import Ry
from Rot3D import Cart2Polar
from Rot3D import Polar2Cart
from Rot3D import Test3Rot
from Rot3D import Ttest2Samples
from Rot3D import FisherMeanTestSameK
from scipy.stats import beta
from FisherDistributions import FisherDistribution2
from Rot3D import AndersonDarling

rejects_lin_b = 0
rejects_lin_c = 0
rejects_spher = 0
rejects_spher_round = 0
ratio = np.inf
# linspace works incorrectly
res_len_values = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
signif_lvls = np.array([0.1, 0.05, 0.025, 0.01])
significance = 0.05
row = 3 - np.where(signif_lvls == significance)[0][0]
largeR = 0
# matrix with critical values for n1=n2=20, with significance lvl 0.1, 0.05, 0.025, 0.01, those values
# decrease bottom to the top, so the first row is the smallest significance
crit_vals_mat66 = np.array([[571, 573, 577, 583, 592, 605, 622, 642, 666, 692, 720, 749, 780, 811],
                            [523, 526, 531, 539, 550, 565, 584, 607, 633, 662, 692, 724, 758, 793],
                            [482, 486, 492, 501, 514, 531, 552, 577, 606, 637, 670, 704, 741, 778],
                            [435, 439, 446, 457, 473, 492, 516, 544, 576, 610, 646, 684, 722, 762]]) * 1 / 1000

crit_vals_mat1010 = np.array([[440, 444, 452, 464, 480, 500, 525, 555, 586, 620, 655, 692, 730, 768],
                              [402, 407, 416, 429, 447, 470, 498, 529, 563, 599, 637, 676, 715, 756],
                              [369, 375, 385, 400, 420, 446, 475, 509, 545, 583, 622, 663, 704, 746],
                              [332, 339, 350, 368, 390, 418, 451, 487, 525, 565, 607, 649, 693, 736]]) * 1 / 1000

crit_vals_mat1515 = np.array([[359, 365, 376, 392, 414, 441, 474, 507, 544, 581, 621, 662, 703, 746],
                              [327, 334, 346, 364, 389, 417, 453, 488, 527, 567, 608, 651, 693, 737],
                              [300, 308, 322, 341, 368, 400, 435, 473, 514, 555, 598, 642, 686, 731],
                              [271, 279, 295, 317, 346, 380, 417, 458, 500, 543, 587, 632, 678, 723]]) * 1 / 1000

crit_vals_mat2020 = np.array([[311, 319, 332, 352, 378, 409, 444, 481, 521, 562, 604, 647, 691, 734],
                              [284, 292, 307, 329, 357, 390, 427, 467, 508, 551, 594, 638, 683, 728],
                              [260, 270, 286, 310, 340, 376, 414, 455, 492, 542, 586, 631, 677, 723],
                              [234, 245, 263, 290, 322, 360, 401, 443, 487, 532, 578, 624, 671, 718]]) * 1 / 1000

crit_vals_mat3030 = np.array([[255, 265, 282, 307, 339, 375, 414, 455, 498, 542, 586, 632, 677, 723],
                              [232, 243, 263, 290, 324, 362, 402, 445, 489, 534, 579, 625, 672, 718],
                              [213, 225, 247, 276, 312, 351, 393, 437, 482, 528, 574, 621, 668, 715],
                              [192, 206, 229, 261, 299, 340, 384, 429, 475, 522, 569, 616, 664, 712]]) * 1 / 1000

trials = 1000
stats = np.zeros((4, trials))
res_len_means1 = np.zeros(trials)
res_len_means2 = np.zeros(trials)
n1 = 30
n2 = 30
crit_dict = {'6_6':crit_vals_mat66, '10_10':crit_vals_mat1010, '15_15':crit_vals_mat1515,
             '20_20':crit_vals_mat2020, '30_30':crit_vals_mat3030}
numstr = str(n1)
crit_vals_mat = [value for key, value in crit_dict.items() if numstr in key][0]

for i in range(trials):
    spher_sample1 = FisherDistribution2(10, n1, 3)
    spher_sample2 = FisherDistribution2(10, n2, 3)
    moved_sample1 = Test3Rot(np.pi / 2 + 15 / 180 * np.pi, np.pi - 15 / 180 * np.pi, spher_sample1)
    moved_sample2 = Test3Rot(np.pi / 2, np.pi, spher_sample2)
    a = FisherMeanTestSameK((moved_sample1, moved_sample2))
    # if np.mod(i,100) == 0:
    #     print (i)

    # n1=n2=20, here compare with azi rejects
    # for +25 degrees the performance is identical at concentration 3
    # there are many more spherical rejects for +15 deg in axi and ele, with concentration 3
    # so not only we need large dispersion but also small difference in true means for linear methods to break down
    # +15 degrees and k=2.5, 130, 4, 62
    # +15 deg k = 3, ratio = 3.4 270, 20, 79
    # +15 deg, k=3.5 still 2.5-3 times more spherical rejects, 403, 62, 134
    # +15 deg, k=4, 377, 165, 189
    # +15 deg, k=4.5, ratio = 0.93, 303, 317, 323
    # +15 deg, k=5.5, ratio = 0.9, 508, 652, 557
    # +15 deg, k=6.5 roughly the same number of rejects for spher azi and ele; 856, 902, 777
    # TBD interpolate the values for z0

    # n1=n2=6, +15 deg performance for concentration parameters in 2-5 is very poor for both methods
    # but linear detects around a dozen cases whereas spherical is 0
    # k = 7 there are 4 spherical rejects and 34 ele, 46 azi
    # k = 8, 27, 87, 82; k = 9, 29, 123, 92
    # at k = 10 ratio = 0.3356, 49, 177, 146
    # k = 15, 318, 463, 436; k = 20, 659, 718, 677; k = 25, 886, 875, 851; k = 50, 1000, 999, 999

    # n1=n2=10, +15 deg
    # k = 2.5, ratio = 5.25, 35, 0, 8; k = 3, ratio =10.7, 75, 3, 7; k = 4, ratio = 12, 99, 8, 8
    # k = 5, ratio = 1.43, 46, 52, 32; k = 6, ratio = 1.42, 50, 24, 35; k = 8, ratio = 0.5, 154, 312, 309
    # k = 10, ratio = 0.8,  453, 567, 555; k = 13, ratio = 0.99, 821, 868, 825

    # n1=n2=15, +15 deg
    # k = 2.5, ratio = 2.7, 57, 0, 21; k = 3, ratio = 5.5, 156, 1, 28; k = 4, ratio = 2.9, 224, 38, 77
    # k = 5, ratio = 0.61, 109, 185, 177; k = 6, ratio = 0.58, 214, 404, 368
    # k = 7, ratio = 0.82, 472, 634, 567; k = 8, ratio = 0.97, 717, 818, 734
    # k = 10, ratio = 1, 967, 966, 939, normality is almost always confirmed here and above

    # n1=n2=30, +15 deg
    # k = 2.5, ratio = 1.96, 289, 27, 147; k = 3, ratio = 1.78, 431, 155, 242; k = 4, ratio = 1.73, 924, 697, 532
    # k = 5, ratio = 1.28, 999, 967, 779; k = 6, ratio = 1.07, 1000, 997, 933; k = 7, ratio = 1.02, 1000, 1000, 981
    # k = 8, ratio = 1, 1000, 1000, 996; k = 10, ratio = 1, 1000, 1000, 1000
    if len(a) == 3:  # i.e if mean_res_len >= 0.75
        moved_polar_coord1 = Cart2Polar(moved_sample1)
        moved_polar_coord2 = Cart2Polar(moved_sample2)
        ele1 = moved_polar_coord1[0, :]
        azi1 = moved_polar_coord1[1, :]
        ele2 = moved_polar_coord2[0, :]
        azi2 = moved_polar_coord2[1, :]
        b = Ttest2Samples(ele1, ele2, 0.1)
        c = Ttest2Samples(azi1, azi2, 0.1)
        if abs(b[0]) > abs(b[1]):
            rejects_lin_b += 1
        if abs(c[0]) > abs(c[1]):
            rejects_lin_c += 1
        if a[0] > a[1]:
            rejects_spher += 1
        teststatistic1 = AndersonDarling(ele1)
        teststatistic2 = AndersonDarling(ele2)
        teststatistic3 = AndersonDarling(azi1)
        teststatistic4 = AndersonDarling(azi2)
        stats[:, i] = np.array([teststatistic1, teststatistic2, teststatistic3, teststatistic4])
        largeR += 1
    else:
        moved_polar_coord1 = Cart2Polar(moved_sample1)
        moved_polar_coord2 = Cart2Polar(moved_sample2)
        ele1 = moved_polar_coord1[0, :]
        azi1 = moved_polar_coord1[1, :]
        ele2 = moved_polar_coord2[0, :]
        azi2 = moved_polar_coord2[1, :]
        b = Ttest2Samples(ele1, ele2, 0.1)
        c = Ttest2Samples(azi1, azi2, 0.1)

        Z = a[0]
        mean_res_len = a[1]
        if mean_res_len > 0.7:
            z0 = crit_vals_mat[row, -1]
        else:
            column = np.digitize(mean_res_len, res_len_values, right=True)  # this gives the bin number
            upper = res_len_values[column]
            lower = res_len_values[column - 1]
            # here we determine which bin the number lies closer two
            a = np.array([abs(lower - mean_res_len), upper - mean_res_len])
            index = np.where(a == min(a))[0][0]
            if index == 1:
                z0 = crit_vals_mat[row, column]
            else:
                z0 = crit_vals_mat[row, column - 1]
        # TBD add code for when R<0.3
        if abs(b[0]) > abs(b[1]):
            rejects_lin_b += 1
        if abs(c[0]) > abs(c[1]):
            rejects_lin_c += 1
        if Z > z0:
            rejects_spher += 1
        teststatistic1 = AndersonDarling(ele1)
        teststatistic2 = AndersonDarling(ele2)
        teststatistic3 = AndersonDarling(azi1)
        teststatistic4 = AndersonDarling(azi2)
        stats[:, i] = np.array([teststatistic1, teststatistic2, teststatistic3, teststatistic4])
if rejects_lin_c != 0:
    ratio = rejects_spher / rejects_lin_c

normals = np.sum(stats < 3, axis=1)
string = "spher rejects = {}, lin rejects ele = {}, lin rejects azi = {}, " \
         "ratio = {}, normals = {}, largeR = {}" \
    .format(rejects_spher, rejects_lin_b, rejects_lin_c, ratio, normals, largeR)
print(string)

# spher_sample1 = FisherDistribution2(2.5, 20, 3)
# spher_sample2 = FisherDistribution2(2.5, 20, 3)
# moved_sample1 = Test3Rot(np.pi/2 + 25/180*np.pi, np.pi/2 - 25/180*np.pi, spher_sample1)
# moved_sample2 = Test3Rot(np.pi/2, np.pi/2, spher_sample2)
# a = FisherMeanTestSameK((moved_sample1, moved_sample2))
# moved_polar_coord1 = Cart2Polar(moved_sample1)
# moved_polar_coord2 = Cart2Polar(moved_sample2)
# ele1 = moved_polar_coord1[0, :]
# azi1 = moved_polar_coord1[1, :]
# ele2 = moved_polar_coord2[0, :]
# azi2 = moved_polar_coord2[1, :]
# b = Ttest2Samples(ele1, ele2, 0.1)
# c = Ttest2Samples(azi1, azi2, 0.1)
# meanele1 = np.mean(ele1)
# meanele2 = np.mean(ele2)
# meanazi1 = np.mean(azi1)
# meanazi2 = np.mean(azi2)

# print(a)
# moved_polar_coord1 = Cart2Polar(moved_sample1)
# moved_polar_coord2 = Cart2Polar(moved_sample2)
# ele1 = moved_polar_coord1[0,:]
# azi1 = moved_polar_coord1[1,:]
# ele2 = moved_polar_coord2[0,:]
# azi2 = moved_polar_coord2[1,:]
# a = Ttest2Samples(ele1, ele2, 0.1)
# b = Ttest2Samples(azi1, azi2, 0.1)
# # x1 = [11,1,0,2,0]
# # x2 = [11,11,5,8,4]
# # # x = Ttest2Samples(x1, x2, 0.1)  # test works correctly, checked with example from a book
# print(a)
# print(b)
# print(x)
# std_ele = np.var(moved_polar_coord[0,:], ddof=1)  # this is probably faster
# compare the azimuth and elevation pair-wise for two populations of points on a sphere
# need to program a t test to test for equality of means then some other test to test for equality of stds
# then on the same data set perform spherical statistics, see in which cases they differ
# then can devise other methods of measuring stds linearly and compare their performance
