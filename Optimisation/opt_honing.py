"""
Performs local optimisation - has not been properly remedied with new version of run_parallel.py
"""
# IMPORTS ####################################################################################################################################################################################
import autograd.numpy as np

from copy import deepcopy

from datetime import datetime

import nlopt

from operator import itemgetter

import pickle

import sys
sys.path.append('../')

from Grating_and_parameters.parameters import D1_ND, Parameters, Initial_bigrating
from opt import FOM_uniform
from Grating_and_parameters.twobox import TwoBox


# EXTRACT OPTIMISATION RESULT ####################################################################################################################################################################################
num_cores = 32
maxfev = 5000

pkl_fname = f'./Data/FOM_3rd_bounds2_optimisation_maxfev{maxfev*num_cores}.pkl'
txt_fname = f'./Data/FOM_3rd_bounds2_optimisation_maxfev{maxfev*num_cores}_curated.txt'
with open(pkl_fname, 'rb') as data_file:
    data = pickle.load(data_file)


## Sort the maxima and maximisers based on value ##
opt_FOMs = data["FOM"]
opt_gratings = data["Optimised grating"]
opt_params = data["Optimised parameters"] 

maxima_and_maximisers = zip(opt_FOMs, opt_params)
maxima_and_gratings = zip(opt_FOMs, opt_gratings)

# Sort all local maxima from largest to smallest
maxima_and_maximisers_sorted = sorted(maxima_and_maximisers, key=itemgetter(0), reverse=True)
opt_gratings_sorted = sorted(maxima_and_gratings, key=itemgetter(0), reverse=True)


# PARAMETERS ####################################################################################################################################################################################
## FIXED PARAMETERS ##
wavelength = 1. # Laser wavelength
angle = 0.
Nx = 100 # Number of grid points
nG = 25 # Number of Fourier components
Qabs = np.inf


# "first"
optimum_number = 0
grating = deepcopy(opt_gratings_sorted[optimum_number][1])
grating.grating_pitch   = 1 / 0.65
grating.params[0] = grating.grating_pitch

grating.Qabs = np.inf # relaxation parameter, should be infinite unless you need to avoid singular matrix at grating cutoffs
wavelength=1.

I0,L,m,c=Parameters()

# FOM parameters
final_speed = 5.
goal = 0.1
return_grad = True

# Parameter bounds
pitch_min = np.round(1.001*wavelength/D1_ND([final_speed/100,0.]),3) # stay away from the cutoff divergences
pitch_max = 1.999 

h1_min = 0. # h1 = grating depth
h1_max = 1.5*pitch_max

box_width_min = 0.
box_width_max = 1.*pitch_max # single box width must be smaller than pitch

box_centre_dist_min = 0.
box_centre_dist_max = 0.5*pitch_max # redundant space if > 0.5*pitch

box_eps_min = 1 # Minimum allowed grating permittivity
box_eps_max = 3.5**2 # Maximum allowed grating permittivity

gaussian_width_min=0.5*L 
gaussian_width_max=5*L

substrate_depth_min = h1_min 
substrate_depth_max = h1_max

substrate_eps_min = box_eps_min 
substrate_eps_max = box_eps_max

# Stopping criteria
xtol_rel = 1e-16    #1e-14
ftol_rel = 1e-16    #1e-14
maxfev = 200


# LOCAL OPTIMISATION ####################################################################################################################################################################################
ndof =  10
init = np.array(grating.params, dtype=np.float64)
bcd_constraint = True
param_bounds = [(pitch_min, pitch_max), (h1_min, h1_max), 
                (box_width_min, box_width_max), (box_width_min, box_width_max),
                (box_centre_dist_min, box_centre_dist_max),
                (box_eps_min, box_eps_max), (box_eps_min, box_eps_max),
                (gaussian_width_min, gaussian_width_max),
                (substrate_depth_min, substrate_depth_max), 
                (substrate_eps_min, substrate_eps_max)]

def objective(params):
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

    return FOM_uniform(grating, final_speed, goal, return_grad)
init_objective = objective(init)[0]

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
    avg_Reigs=grating_check.average_real_eigs(final_speed,goal,return_eigs=False)

    MAX = np.max(avg_Reigs) 
    if MAX ==0:
        condition = 1
    else:
        condition = MAX
            
    return condition

ctrl = 0 # count steps
steps = [] # record steps 
def fun_nlopt(params,gradn):
    global ctrl, steps
    
    y, dy = objective(params)
    if gradn.size > 0:
        gradn[:] = dy

    grating_pitch, grating_depth , box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, gaussian_width, substrate_depth, substrate_eps = params

    step = (ctrl, y, params)
    steps.append(step)
    step_result = f"Step: {ctrl}, FOM: {y}"
    print(step_result)
    print("params:")
    print("grating_pitch:   ", grating_pitch)
    print("grating_depth:   ", grating_depth)
    print("box1_width:      ", box1_width)
    print("box2_width:      ", box2_width)
    print("box_centre_dist: ", box_centre_dist)
    print("box1_eps:        ", box1_eps)
    print("box2_eps:        ", box2_eps)
    print("gaussian_width:  ", gaussian_width)
    print("substrate_depth: ", substrate_depth)
    print("substrate_eps:   ", substrate_eps)
    print("\n")

    ctrl += 1
    return y


## Optimisation ##
local_opt = nlopt.opt(nlopt.LD_MMA, ndof)

lb = [bounds[0] for bounds in param_bounds]
ub = [bounds[1] for bounds in param_bounds]
local_opt.set_lower_bounds(lb)
local_opt.set_upper_bounds(ub)

local_opt.set_xtol_rel(xtol_rel)
local_opt.set_ftol_rel(ftol_rel)
local_opt.set_maxeval(maxfev)

# Set options for local optimiser
if bcd_constraint:
    local_opt.add_inequality_constraint(bcd_not_redundant)
local_opt.add_inequality_constraint(box_clips_cell_edge)
local_opt.add_inequality_constraint(box_gaps_non_zero)
local_opt.add_inequality_constraint(avg_eig_all_negative)

local_opt.set_max_objective(fun_nlopt)

local_opt.set_param( "verbosity", 0 )

opt_params = local_opt.optimize(init)
optimum = local_opt.last_optimum_value()

## Print final result
grating_pitch, grating_depth , box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, gaussian_width, substrate_depth, substrate_eps = opt_params
print("Final optimised parameters:")
print("grating_pitch:   ", grating_pitch)
print("grating_depth:   ", grating_depth)
print("box1_width:      ", box1_width)
print("box2_width:      ", box2_width)
print("box_centre_dist: ", box_centre_dist)
print("box1_eps:        ", box1_eps)
print("box2_eps:        ", box2_eps)
print("gaussian_width:  ", gaussian_width)
print("substrate_depth: ", substrate_depth)
print("substrate_eps:   ", substrate_eps)

## Converting parameter dicts to strings ##
# Results
init_line = repr(init)
opt_params_line = repr(opt_params)

# Fixed parameters
fixed_params_dict = {'wavelength': grating.wavelength,
                    'angle': grating.angle,
                    'Nx': grating.Nx, 'nG': grating.nG, 'Qabs': grating.Qabs}
fixed_params_line = str(fixed_params_dict)
FOM_params_dict = {'final_speed': final_speed, 'goal': goal}
FOM_params_line = str(FOM_params_dict)

# Bounded parameters
bounds_dict = {'bounds': param_bounds}
bounds_line = str(bounds_dict)

# Optimiser options
LO_dict = {'xtol_rel': f"{xtol_rel:.1E}", 'ftol_rel': f"{ftol_rel:.1E}", 'maxfev': maxfev}
LO_line = str(LO_dict)

# Date and time at beginning of run
time_at_execution = str(datetime.now())

# Strings to write to file
init_lines = ["\n\n------------------------------------------------------------------------------------------------------------------------------------\n"
                , f"LO Honing \n"
                , f"Date & time         | {time_at_execution}\n"
                ,  "\n"
                , f"Fixed parameters    | {fixed_params_line}\n"
                , f"FOM parameters      | {FOM_params_line}\n"
                , f"Bounds              | {bounds_line}\n"
                ,  "\n"
                , f"LO options          | {LO_line}\n"
                , f"Initial guess       | ({init_objective}, ({init[0]}, {init[1]}, {init[2]}, {init[3]}, {init[4]}, {init[5]}, {init[6]}, {init[7]}, {init[8]}, {init[9]}))\n"
                , "------------------------------------------------------------------------------------------------------------------------------------\n"]

step_lines = []
for step in steps:
    ctrl, y, params = step
    step_line = f"Step: {ctrl}, FOM: {y}\n"
    step_lines.append(step_line)

result_lines = ["------------------------------------------------------------------------------------------------------------------------------------\n"
                , f"LO (Max, maximiser) | ({optimum}, {grating_pitch}, {grating_depth}, {box1_width}, {box2_width}, {box_centre_dist}, {box1_eps}, {box2_eps}, {gaussian_width}, {substrate_depth}, {substrate_eps})\n"
                , "------------------------------------------------------------------------------------------------------------------------------------\n"]

txt_fname = f'./Data/FOM_LO_maxfev{maxfev}.txt' # save to
with open(txt_fname, "a") as result_file:
    result_file.writelines(init_lines)
    result_file.writelines(step_lines)
    result_file.writelines(result_lines)