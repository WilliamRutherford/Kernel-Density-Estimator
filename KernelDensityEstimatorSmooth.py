import math
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.stats import norm
from KernelDensityEstimator import KernelDensityEstimator, gaussian_fn

'''
A Kernel Density Estimator where no bandwidth is provided; we simply choose the bandwidth which is the most 'smooth'
'''
def smoothGaussEstimator(data_pts : npt.ArrayLike, divs : int = 100, bounds_factor : float = 0.25, bounds_abs = None, input_pts = None):
    # calculation of average distance between consecutive points
    sorted_pts = np.sort(data_pts)
    pt_distances = np.abs(sorted_pts[1:] - sorted_pts[:-1])
    pt_dist_avg = np.mean(pt_distances)
    # whatever that is, use it as as the standard deviation
    std_dev = pt_dist_avg * 0.75
    result = KernelDensityEstimator(data_pts = data_pts, bandwidth = std_dev, divs = divs, bounds_factor = bounds_factor, bounds_abs = bounds_abs, input_pts = input_pts)
    return result



if(__name__ == "__main__"):
    bimodal_test_data = np.concatenate((np.random.normal(loc = -3, scale = 0.7, size = 10), np.random.normal(loc = 5, scale = 1.15, size = 10)))
    test_gauss = smoothGaussEstimator(data_pts = bimodal_test_data)
    
    # Plot the results
    eval_pts, relative_density, std_dev = test_gauss
    print("Standard deviation used: ", std_dev)
    plt.plot(eval_pts, relative_density, label="Estimated Density")
    plt.title("Kernel Density Estimation")
    plt.xlabel("Data Points")
    plt.ylabel("Density")
    plt.legend()
    plt.show()