"""
Contains the radiation pressure cross-sections and the functions that optimise a structure with given search-space bounds.
"""


# IMPORTS ########################################################################################################################
import adaptive as adp
from autograd import grad

import numpy as np
import nlopt

import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

import sys
sys.path.append("../")

import random

from typing import Callable

from Grating_and_parameters.parameters import Parameters, D1_ND, Initial_bigrating
from Grating_and_parameters.twobox import TwoBox

I0, L, m, c = Parameters()

# FUNCTIONS ########################################################################################################################

def FD(grating: TwoBox) -> float:
    """
    Calculate the grating single-wavelength figure of merit FD.

    Parameters
    ----------
    grating :           TwoBox instance containing the grating parameters
    """
    
    return grating.FoM(I0, grad_method = "finite")

def FD_params_func(grating, params):
    grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, gaussian_width, substrate_depth, substrate_eps = params
    
    grating.grating_pitch = grating_pitch
    grating.grating_depth = grating_depth
    grating.box1_width = box1_width
    grating.box2_width = box2_width
    grating.box_centre_dist = box_centre_dist
    grating.box1_eps = box1_eps
    grating.box2_eps = box2_eps
    
    grating.gaussian_width=gaussian_width
    grating.substrate_depth = substrate_depth
    grating.substrate_eps = substrate_eps

    return FD(grating)

FD_grad = grad(FD_params_func, argnum=1)

def FOM_uniform(grating: TwoBox, final_speed: float=20., goal: float=0.1, return_grad: bool=True) -> float:
    """
    Calculate wavelength expectation of FD FOM (figure of merit) for the given grating over a fixed wavelength range.
    Assumes a uniform probability distribution for wavelength.

    Parameters
    ----------
    grating     :   TwoBox instance containing the grating parameters
    final_speed :   Final sail speed as percentage of light speed
    goal        :   Stopping goal for wavelength integration passed to adaptive runner. If int, use npoints_goal; if float, use loss_goal.
    return_grad :   Return [FOM, FOM gradient]
    """
    laser_wavelength = grating.wavelength # copy the starting wavelength
    Doppler = D1_ND([final_speed/100,0])
    l_min = 1 # l = grating frame wavelength normalised to laser frame wavelength
    l_max = l_min/Doppler    
    l_range = (l_min, l_max)

    # Perturbation probability density function (PDF)
    PDF_unif = 1/(l_max-l_min)
    
    # Define a one argument function to pass to learner
    def weighted_FD(l):
        grating.wavelength = l*laser_wavelength
        return PDF_unif*FD(grating)
    
    # Adaptive sample FD
    FD_learner = adp.Learner1D(weighted_FD, bounds=l_range)
    if isinstance(goal, int):
        FD_runner = adp.runner.simple(FD_learner, npoints_goal=goal)
    elif isinstance(goal, float):
        FD_runner = adp.runner.simple(FD_learner, loss_goal=goal)
    else: 
        raise ValueError("Sampling goal type not recognised. Must be int for npoints_goal or float for loss_goal.")
    
    FD_data = FD_learner.to_numpy()
    l_vals = FD_data[:,0]
    weighted_FDs = FD_data[:,1]
    
    FOM = np.trapz(weighted_FDs,l_vals)

    if return_grad:
        """
        Should return FOM (average FD over wavelength) gradient at the given grating parameters.

        Implemented by first calculating the gradient at the grating parameters then averaging the gradient over wavelength
        """
        
        # Need to copy the following immutable parameters to pass to FD_grad, otherwise get UFuncTypeError
        grating_pitch = grating.grating_pitch
        grating_depth = grating.grating_depth
        box1_width = grating.box1_width
        box2_width = grating.box2_width
        box_centre_dist = grating.box_centre_dist
        box1_eps = grating.box1_eps
        box2_eps = grating.box2_eps
        
        gaussian_width=grating.gaussian_width
        substrate_depth = grating.substrate_depth
        substrate_eps = grating.substrate_eps

        params = [grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps,
                  gaussian_width, substrate_depth, substrate_eps]
        
        # Define a one argument function to pass to learner
        def weighted_FD_grad(l):
            grating.wavelength = l*laser_wavelength
            return PDF_unif*np.array(FD_grad(grating, params))

        # Adaptive sample FD_grad
        FD_grad_learner = adp.Learner1D(weighted_FD_grad, bounds=l_range)

        if isinstance(goal, int):
            FD_grad_runner = adp.runner.simple(FD_grad_learner, npoints_goal=goal)
        elif isinstance(goal, float):
            FD_grad_runner = adp.runner.simple(FD_grad_learner, loss_goal=goal)
        
        FD_grad_data = FD_grad_learner.to_numpy()
        l_vals = FD_grad_data[:,0]
        weighted_FD_grads = FD_grad_data[:,1:]
        
        FOM_grad = np.trapz(weighted_FD_grads,l_vals, axis=0)

        grating.wavelength = laser_wavelength # restore user-initialised wavelength
        return [FOM, FOM_grad] 
    else:
        grating.wavelength = laser_wavelength # restore user-initialised wavelength
        return FOM

def global_optimise(objective, 
                    sampling_method: str="sobol", seed: int=0, n_sample: int=8, maxfev: int=32000,
                    xtol_rel: float=1e-4, ftol_rel: float=1e-8, param_bounds: list=[]):
    """
    Global optimise the twobox on a single CPU core using MLSL global with MMA local.

    Parameters
    ----------
    objective       :   Objective function to optimise. Objective must return (value, gradient)
    sampling_method :   "sobol" or "random" initial point sampling
    seed            :   Seed for initial random parameter space sample and grating_depth samples
    n_sample        :   Number of points for initial sample (per dimension?)
    maxfev          :   Maximum function evaluations per core
    xtol_rel        :   Relative position tolerance for MMA
    ftol_rel        :   Relative objective tolerance for MMA
    param_bounds    :   List of tuples, each containing lower and upper bounds on a parameter
    """
    grating_pitch, _ , box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, gaussian_width, substrate_depth, substrate_eps =Initial_bigrating()
    
    h1_min, h1_max = param_bounds[1]
    ndof = 10    
    h1_start = random.uniform(h1_min,h1_max)
    init = [grating_pitch, h1_start, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps,
            gaussian_width, substrate_depth, substrate_eps] 
    bcd_constraint = True 

    # Obey nlopt syntax for objective and constraint functions 
    def fun_nlopt(params,gradn):
        y, dy = objective(params)
        if gradn.size > 0:
            # Even for gradient methods, in some calls gradn will be empty []
            gradn[:] = dy
        return y

    # Constraints to add - must be of form h(x)<=0
    def bcd_not_redundant(params,gradn): 
        """
        Unit cell periodicity means boxes with separation >0.5*Lam are
        equivalent to boxes with separation <0.5*Lam
        """
        Lam, _, _, _, bcd, _, _, _, _, _ = params
        # condition = bcd - 0.5*Lam
        condition = np.abs(bcd - 0.25*Lam) - 0.25*Lam 
        return condition

    def box_gaps_non_zero(params,gradn):
        """
        Want a bigrating, so make the distance between two unit cells bigger than zero
        """
        _, _, w1, w2, bcd, _, _, _, _, _ = params
        condition= (w1+w2)/2 - bcd 
        return condition

    def box_clips_cell_edge(params,gradn):
        """
        Gradients become expensive to compute if the boxes are cutoff at the edge of the unit cell.
        """
        Lam, _, w1, w2, bcd, _, _, _, _, _ = params
        condition = (w1+w2)/2 + bcd - 0.98*Lam
        return condition
    
    def avg_eig_all_negative(params,gradn):
        """
        Ensure on average (over wavelength), Re(eigenvalues) are all negative
        """
        from run_parallel import final_speed, goal, angle, Nx,nG,Qabs
        # Build grating from parameters
        
        Lam, h1, w1, w2, bcd, eps1, eps2, w, t, s_eps = params
        grating_check=TwoBox(Lam,h1,w1,w2,bcd,eps1,eps2,w,t,s_eps,
                             1,angle,Nx,nG,Qabs)
        avg_Reigs=grating_check.average_real_eigs(final_speed,goal,return_eigs=False, I=I0)

        MAX = np.max(avg_Reigs) 
        if MAX ==0:
            condition = 1
        else:
            condition = MAX
                
        return condition
    
    # Choose GO and LO
    if sampling_method == 'sobol':
        global_opt = nlopt.opt(nlopt.G_MLSL_LDS, ndof)
    elif sampling_method == 'random':
        global_opt = nlopt.opt(nlopt.G_MLSL, ndof)
    else:
        global_opt = nlopt.opt(nlopt.G_MLSL_LDS, ndof)
    local_opt = nlopt.opt(nlopt.LD_MMA, ndof)

    # Set LDS and initial grating_depth seed
    nlopt.srand(seed) 
    random.seed(seed) 

    # Set options for optimiser
    global_opt.set_population(n_sample) # initial sampling points

    # Set options for local optimiser
    if bcd_constraint:
        local_opt.add_inequality_constraint(bcd_not_redundant)
    local_opt.add_inequality_constraint(box_clips_cell_edge)
    local_opt.add_inequality_constraint(box_gaps_non_zero)
    local_opt.add_inequality_constraint(avg_eig_all_negative)

    local_opt.set_xtol_rel(xtol_rel)
    local_opt.set_ftol_rel(ftol_rel)
    global_opt.set_local_optimizer(local_opt)

    # Set objective function
    global_opt.set_max_objective(fun_nlopt)
    global_opt.set_maxeval(maxfev)
    
    lb = [bounds[0] for bounds in param_bounds]
    ub = [bounds[1] for bounds in param_bounds]
    global_opt.set_lower_bounds(lb)
    global_opt.set_upper_bounds(ub)

    import traceback

    try:
        opt_params = global_opt.optimize(init)
        optimum = objective(opt_params)[0]
        print("Success on starting bounds: ", param_bounds[1])
        is_optimum = True
        return (optimum, opt_params, is_optimum)
    except:
        print("Failed on starting bounds: ",param_bounds[1])
        traceback.format_exc()
        fake_FOM = - np.inf
        fake_params = init
        is_optimum = False
        return (fake_FOM, fake_params, is_optimum)

    