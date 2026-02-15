import math
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.stats import norm

def GenericGaussEstimator(data_pts : npt.ArrayLike, fn, divs : int = 100, bounds_factor : float = 0.25, bounds_abs = None, input_pts = None):
    
    if(input_pts is None):
        if(bounds_abs is None):
            data_range = np.max(data_pts) - np.min(data_pts)
            lower_bnd = np.min(data_pts) - bounds_factor * data_range
            upper_bnd = np.max(data_pts) + bounds_factor * data_range
        else:
            lower_bnd = bounds_abs[0]
            upper_bnd = bounds_abs[1]
        eval_pts = np.linspace(lower_bnd, upper_bnd, divs)
    else:
        eval_pts = input_pts

    # calculation of average distance between consecutive points
    sorted_pts = np.sort(data_pts)
    pt_distances = np.abs(sorted_pts[1:] - sorted_pts[:-1])
    pt_dist_avg = np.mean(pt_distances)
    # whatever that is, use it as as the standard deviation
    std_dev = pt_dist_avg * 0.75



    # let n be the number of data_pts, or len(data_pts)
    
    # data has shape (n,)
    # data_stretched has shape (n, divs) where data_stretched[:, i] = data_pts
    data_stretched = np.column_stack(divs * (data_pts,))
    # eval has shape (divs,)
    # eval_stretched has shape (n, divs) where eval_stretched[i, :] = eval_pts
    eval_stretched = np.tile(eval_pts, (len(data_pts), 1))
    
    dist_diff = data_stretched - eval_stretched
    # For each offset, replace with the probability from the gaussian pdf. 
    norm_func = fn
    #vec_norm_func = np.vectorize(norm_func)
    # Generate the probability distribution for each offset
    ind_density = norm_func(dist_diff)
    # Sum up each column, representing the total density for each point sampled. 
    tot_density = np.sum(ind_density, axis = 0)
    # Then, we make it relative to the whole area; giving us a probability density estimation (the area underneath is 1)
    relative_density = tot_density / np.sum(tot_density)
    return eval_pts, relative_density

if(__name__ == "__main__"):
    fn = (lambda x: (abs(x) <= 1) * 0.5)
    bimodal_test_data = np.concatenate((np.random.normal(loc = -3, scale = 0.7, size = 10), np.random.normal(loc = 5, scale = 1.15, size = 10)))
    test_gauss = GenericGaussEstimator(data_pts = bimodal_test_data, fn = fn)
    
    # Plot the results
    eval_pts, relative_density = test_gauss
    plt.plot(eval_pts, relative_density, label="Estimated Density")
    plt.title("Kernel Density Estimation")
    plt.xlabel("Data Points")
    plt.ylabel("Density")
    plt.legend()
    plt.show()