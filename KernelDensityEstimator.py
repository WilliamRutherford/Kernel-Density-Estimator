import math
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.stats import norm

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
def discrete_estimator(data_pts : npt.ArrayLike, bandwidth : float = 1.0, divs : int = 100, bounds_factor : float = 0.25, bounds_abs = None, input_pts = None):
    # For each data point, we have a normal distribution offset by it's value
    # We then plug in a bunch of values, and sum over the result of the PDF
    # For each point we evaluate, we only care about the distance between it and another point (the supposed 'center') meaning we can do pdf(other_pt - curr_pt)
    if(bounds_abs is None):
        data_range = np.max(data_pts) - np.min(data_pts)
        lower_bnd = np.min(data_pts) - bounds_factor * data_range
        upper_bnd = np.max(data_pts) + bounds_factor * data_range
    else:
        lower_bnd = bounds_abs[0]
        upper_bnd = bounds_abs[1]
    if(input_pts is None):
        eval_pts = np.linspace(lower_bnd, upper_bnd, divs)
    else:
        eval_pts = input_pts
    # let n be the number of data_pts, or len(data_pts)
    
    # data has shape (n,)
    # data_stretched has shape (n, divs) where data_stretched[:, i] = data_pts
    data_stretched = np.column_stack(divs * (data_pts,))
    # eval has shape (divs,)
    # eval_stretched has shape (n, divs) where eval_stretched[i, :] = eval_pts
    eval_stretched = np.vstack(len(data_pts) * (eval_pts,))
    
    dist_diff = data_stretched - eval_stretched
    # For each offset, replace with the probability from the gaussian pdf. 
    norm_func = gaussian_fn(center = 0.0, std_dev = bandwidth) 
    # Generate the probability distribution for each offset
    ind_density = norm_func(dist_diff)
    # Sum up each column, representing the total density for each point sampled. 
    tot_density = np.sum(ind_density, axis = 0)
    # Then, we make it relative to the whole area; giving us a probability density estimation (the area underneath is 1)
    relative_density = tot_density / np.sum(tot_density)
    return eval_pts, relative_density

'''
A Kernel Density Estimator which can be constantly updated with new data.

Constructor inputs:
starting_data (Arraylike)  : the 1D datapoints we start with. 
starting_bandwidth (float) : the standard deviation of each component Gaussian normal distribution
divs (int) : the number of equally spaced points to evaluate for our estimator
eval_pts (Arraylike) : Optional; a set array of points to evaluate our estimator at. Useful if we will always evaluate at the same points.
bounds (tuple) : Optional; the upper and lower bound for where we evaluate our estimator. If provided, the bounds will not be automatically updated. 
bnds_factor (float) : Optional; if we have variable upper and lower bounds, this represents the fraction of our data's range we include below the minimum and above the maximum. 
'''
class UpdatingKernelDensityEstimator:
    
    def __init__(starting_data : npt.ArrayLike = np.array([], dtype='float64'), starting_bandwidth : float = 1.0, divs : int = 100, eval_pts = None, bounds : npt.ArrayLike = None, bnds_factor : float = 0.25):
        self.data_pts = starting_data.copy()
        self.bandwidth = starting_bandwidth
        self.divs = divs
        self.eval_pts = eval_pts
        self.bnds_factor = bnds_factor
        self.bounds = bounds
        
        self.const_eval_pts = (eval_pts is not None)
        self.const_bounds = (bounds is not None)
        
        self.tot_density = np.array([], dtype='float64')
        self.integral_area = 0.0
        self.norm_fn = gaussian_fn(center = 0.0, std_dev = self.bandwidth)
        
        if(len(starting_data) != 0):
            pass
        
    def add_data(data_pts : npt.ArrayLike, full_update : bool = False):
        # If the provided datapoints are in a matrix, we flatten it
        if(len(data_pts.shape) > 1):
            in_data = data_pts.flatten()
        else:
            in_data = data_pts
        # Concatenate our current data with this new data    
        self.data_pts = np.concatenate((self.data_pts, in_data))
        # New datapoints means new probability densities to calculate. We would prefer to do this for JUST the new data, instead of redoing the entire array. 
        # We do a full update when told, or when crucial variables are yet to be assigned. 
        if(full_update or (self.eval_pts is None)):
            pass
        else:
            # Without full update, we can calculate the difference between each eval point and each new datapoint, and attach it to self.dist_diff with np.vstack
            in_data_stretched = np.column_stack(len(eval_pts) * (in_data,))
            eval_stretched = np.vstack(len(in_data) * (eval_pts,))
            dist_diff = in_data_stretched - eval_stretched
            ind_density = self.norm_fn(dist_diff)
            new_tot_density = np.sum(ind_density, axis = 0)
            self.tot_density += new_tot_density
            self.integral_area = np.sum(self.tot_density)
            return self.tot_density / self.integral_area
    
    '''
    Return the kernel density estimation at each evaluation point
    '''        
    def get_estimator():
        return self.tot_density / self.integral_area

bimodal_test_data = np.concatenate((np.random.normal(loc = -3, scale = 0.7, size = 10), np.random.normal(loc = 5, scale = 1.15, size = 10)))