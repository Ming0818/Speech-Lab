from proto2 import *
from prondict import *
import numpy as np
import matplotlib.pyplot as plt

if __name__== "__main__":

    N = 15
    D = 2
    M = 3

    gamma = np.zeros((N,M))
    gamma[:4,0] = 1.0
    gamma[4:10,1] = 1.0
    gamma[10:,2] = 1.0

    X = np.random.rand(N,D)

    means, covars = updateMeanAndVar(X,np.log(gamma), varianceFloor = 0.0)

    check_m0 = means[0,:] -np.mean(X[:4],axis=0)
    check_m1 = means[1,:] -np.mean(X[4:10],axis=0)
    check_m2 = means[2,:] -np.mean(X[10:],axis=0)
    check_c0 = covars[0, :] - np.var(X[:4], axis=0)
    check_c1 = covars[1, :] - np.var(X[4:10], axis=0)
    check_c2 = covars[2, :] - np.var(X[10:], axis=0)

    print(check_m0)
    print(check_m1)
    print(check_m2)
    print(check_c0)
    print(check_c1)
    print(check_c2)