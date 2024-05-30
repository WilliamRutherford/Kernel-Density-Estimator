import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

'''
A single Gaussian Normal Distribution density function. 
Should use numpy methods and return a function, so it can operate on a whole array of elements. 
'''
def gaussian(center = 0.0, std_dev = 1.0):
    return (lambda x : 1 / (std_dev * math.sqrt(2 * math.pi)) * np.exp((-1/2) * ( (x - center) /std_dev)**2)) 

'''
Given a set of data points, create 'divs' linearly spaced points, and generate a Kernel Density Estimation. 
bandwidth : the influence of each point, equivalent to the standard deviation of each normal distribution
divs : the number of x values to calculate the density estimation of. 
bounds_factor: The amount to spread the minimum and maximum point, relative to the total range. 
bounds_abs : (Optional) the specific upper and lower bounds of where we evaluate. Used when we are fitting the format of other data. 
'''
def discrete_estimator(data_pts, bandwidth = 1, divs = 100, bounds_factor = 0.25, bounds_abs = None):
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
    eval_pts = np.linspace(lower_bnd, upper_bnd, divs)
    # let n be the number of data_pts, or len(data_pts)
    
    # data has shape (n,)
    # data_stretched has shape (n, divs) where data_stretched[:, i] = data_pts
    data_stretched = np.column_stack(divs * (data_pts,))
    # eval has shape (divs,)
    # eval_stretched has shape (n, divs) where eval_stretched[i, :] = eval_pts
    eval_stretched = np.vstack(len(data_pts) * (eval_pts,))
    
    dist_diff = data_stretched - eval_stretched
    # For each offset, replace with the probability from the gaussian pdf. 
    norm_func = gaussian(center = 0.0, std_dev = bandwidth) 
    # Generate the probability distribution for each offset
    ind_density = norm_func(dist_diff)
    # Sum up each column, representing the total density for each point sampled. 
    tot_density = np.sum(ind_density, axis = 0)
    # Then, we make it relative to the whole area; giving us a probability density estimation (the area underneath is 1)
    rel_density = tot_density / np.sum(tot_density)
    return eval_pts, rel_density

bimodal_test_data = np.concatenate((np.random.normal(loc = -3, scale = 0.7, size = 10), np.random.normal(loc = 5, scale = 1.15, size = 10)))