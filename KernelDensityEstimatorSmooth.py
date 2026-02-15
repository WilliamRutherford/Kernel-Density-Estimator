import math
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.stats import norm
from KernelDensityEstimator import KernelDensityEstimator, gaussian_fn
import time

'''
A Kernel Density Estimator where no bandwidth is provided; we simply choose the bandwidth which is the most 'smooth'
'''
def smoothGaussEstimator(data_pts : npt.ArrayLike, divs : int = 100, bounds_factor : float = 0.25, bounds_abs = None, input_pts = None, timing: bool = False):
    if timing:
        start_time = time.time()
    
    # calculation of average distance between consecutive points
    sorted_pts = np.sort(data_pts)
    pt_distances = np.abs(sorted_pts[1:] - sorted_pts[:-1])
    pt_dist_avg = np.mean(pt_distances)
    std_dev = pt_dist_avg * 0.75

    if timing:
        print(f"Step 1 (Calculate std_dev): {time.time() - start_time:.4f} seconds")
        start_time = time.time()

    result = KernelDensityEstimator(data_pts=data_pts, bandwidth=std_dev, divs=divs, bounds_factor=bounds_factor, bounds_abs=bounds_abs, input_pts=input_pts)

    if timing:
        print(f"Step 2 (KernelDensityEstimator): {time.time() - start_time:.4f} seconds")
    
    return result



if(__name__ == "__main__"):
    bimodal_test_data = np.concatenate((np.random.normal(loc=-3, scale=0.7, size=10), np.random.normal(loc=5, scale=1.15, size=10)))
    test_gauss = smoothGaussEstimator(data_pts=bimodal_test_data, timing=True)
    
    # Plot the results
    eval_pts, relative_density, std_dev = test_gauss
    print("Standard deviation used: ", std_dev)
    plt.plot(eval_pts, relative_density, label="Estimated Density")
    plt.title("Kernel Density Estimation")
    plt.xlabel("Data Points")
    plt.ylabel("Density")
    plt.legend()
    plt.show()