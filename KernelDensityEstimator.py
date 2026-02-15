import math
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.stats import norm
from KernelDensityEstimatorGeneric import GenericGaussEstimator

'''
A single Gaussian Normal Distribution density function. 
Should use numpy methods and return a function, so it can operate on a whole array of elements. 
'''
def gaussian_fn(center = 0.0, std_dev = 1.0):
    return (lambda x : 1 / (std_dev * math.sqrt(2 * math.pi)) * np.exp((-1/2) * ( (x - center) /std_dev)**2)) 

'''
Given a set of data points, create 'divs' linearly spaced points, and generate a Kernel Density Estimation. 
Arguments:
bandwidth : the influence of each point, equivalent to the standard deviation of each component normal distribution
divs : the number of x values to calculate the density estimation for 
bounds_factor: The amount to spread the minimum and maximum point, relative to the total range 
bounds_abs : (Optional) the specific upper and lower bounds of where we evaluate. Used when we are fitting the format of other data ranges 
input_ots : (Optional) the specific values to calculate the probability density at. Used when we are evaluating the same points on multiple occasions

Returns:
evaluation_points, density_estimation
evaluation_pts has shape (divs,) representing the points it was evaluated at
density_estimation has shape (divs,) representing the relative probability density at each evaluated point
'''
def KernelDensityEstimator(data_pts : npt.ArrayLike, bandwidth : float = 1.0, divs : int = 100, bounds_factor : float = 0.25, bounds_abs = None, input_pts = None):
    pass
    fn = gaussian_fn(center = 0.0, std_dev = bandwidth)
    result = GenericGaussEstimator(data_pts = data_pts, fn = fn, divs = divs, bounds_factor = bounds_factor, bounds_abs = bounds_abs, input_pts = input_pts)
    return result + (bandwidth,)

if(__name__ == "__main__"):
    bimodal_test_data = np.concatenate((np.random.normal(loc = -3, scale = 0.7, size = 10), np.random.normal(loc = 5, scale = 1.15, size = 10)))
    test_gauss = KernelDensityEstimator(data_pts = bimodal_test_data)
    
    # Plot the results
    eval_pts, relative_density, std_dev = test_gauss
    print("Standard deviation used: ", std_dev)
    plt.plot(eval_pts, relative_density, label="Estimated Density")
    plt.title("Kernel Density Estimation")
    plt.xlabel("Data Points")
    plt.ylabel("Density")
    plt.legend()
    plt.show()