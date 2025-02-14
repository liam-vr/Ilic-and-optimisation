"""
Store 
"""
import numpy as np

I0=10e9         # intensity - 10 Gw/m^2
L=10            # grating length (total) (10 m^2)
m=1/1000        # mass (1g)
c=299792458     # speed of light

def Parameters():
    """
    ## Outputs
    I0: intensity defined in parameters.py
    L: total lengtgh of bigrating
    m: mass of bigrating
    c: speed of light
    """
    return I0, L, m, c

## Optimised grating
def gaussian_width(grating_type):
    """
    ## Inputs
    grating_type: 'Ilic', 'Ilic-damp'
    ## Outputs 
    Returns the (NOT normalised by L) gaussian beam width 
    """
    if grating_type=="Ilic":
        return 2 * L
    if grating_type=="Ilic-damp":
        return 2.7180049942915896 * L
    if grating_type=="Second":
        return 33.916288616522735

#######################################################
## Parameters only used for optimisation
from Grating_and_parameters.SR_functions import D1_ND, gamma_ND

## Global Optimisation parameters ##
wavelength = 1. # Laser wavelength
angle = 0.
Nx = 100 # Number of grid points
nG = 25 # 25 # Number of Fourier components
# relaxation parameter, should be infinite unless you need to avoid singular matrix at grating cutoffs
# Also, optimiser finds large magnitude, noisy rNeg1 when Qabs = np.inf 
Qabs = 1e7  # 1e7

## FoM parameters ##
goal = 0.1 # Stopping criteria for adaptive sampling in the FOM (float is loss_goal, int is npoints_goal)
final_speed = 5 # percentage of c  # 20
return_grad = True # Return FOM and gradient of FOM

## Global Optimisation bounds ##
## Parameter bounds
pitch_min = np.round(1.001*1/D1_ND([final_speed/100,0.]),3) # stay away from the cutoff divergences
pitch_max = np.round( 2/( 1 + np.sin(20*(np.pi/180)) ), 3) 

h1_min = 0.01 # h1 = grating depth
h1_max = 1.5 * pitch_max

box_width_min = 0.
box_width_max = 1.*pitch_max # single box width must be smaller than pitch

box_centre_dist_min = 0.
box_centre_dist_max = 0.5*pitch_max# redundant space if > 0.5*pitch

box_eps_min = 1.5**2 # Minimum allowed grating permittivity
box_eps_max = 3.5**2 # Maximum allowed grating permittivity

gaussian_width_min=0.5*L 
gaussian_width_max=5*L

substrate_depth_min = h1_min 
substrate_depth_max = 1.5 * pitch_max # h1_max

substrate_eps_min = box_eps_min 
substrate_eps_max = box_eps_max

param_bounds = [(pitch_min, pitch_max), (h1_min, h1_max), 
            (box_width_min, box_width_max), (box_width_min, box_width_max),
            (box_centre_dist_min, box_centre_dist_max),
            (box_eps_min, box_eps_max), (box_eps_min, box_eps_max),
            (gaussian_width_min, gaussian_width_max),
            (substrate_depth_min, substrate_depth_max), 
            (substrate_eps_min, substrate_eps_max)]

## Optimisation parameters
def opt_Parameters():
    return wavelength, angle, Nx, nG, Qabs, goal, final_speed, return_grad

def Bounds():
    return h1_min, h1_max, param_bounds

## Initial grating
Start="middle"

if Start=="Ilic":
    wavelength=1.5 /D1_ND(2/100)

    grating_pitch=1.8 / wavelength
    grating_depth=0.5 / wavelength
    box1_width=0.15 * grating_pitch
    box2_width=0.35 * grating_pitch
    box_centre_dist=0.60 * grating_pitch
    box1_eps = 3.5**2 
    box2_eps = 3.5**2
    gaussian_width_value=2 * L
    substrate_depth=0.5 / wavelength
    substrate_eps=1.45**2

    wavelength=1

    grating_pitch   = np.float64(grating_pitch)
    grating_depth   = np.float64(grating_depth)
    box1_width      = np.float64(box1_width)
    box2_width      = np.float64(box2_width)
    box_centre_dist = np.float64(box_centre_dist)
    box1_eps        = np.float64(box1_eps)
    box2_eps        = np.float64(box2_eps)
    gaussian_width  = np.float64(gaussian_width)
    substrate_depth = np.float64(substrate_depth)
    substrate_eps   = np.float64(substrate_eps)

if Start=="other":
    grating_pitch   = np.float64(1.3181953910424888)
    grating_depth   = np.float64(0.9981573495961377)
    box1_width      = np.float64(0.3925889472885004)
    box2_width      = np.float64(0.10992496313652637)
    box_centre_dist = np.float64(0.46325492497782883)
    box1_eps        = np.float64(2.3183598380520047)
    box2_eps        = np.float64(2.6699220192321977)
    gaussian_width_value  = np.float64(25.03906252365165)
    substrate_depth = np.float64(2.9915278797460836)
    substrate_eps   = np.float64(2.3183620304237067)

if Start=="middle":
    def average(x,y): return (x+y)/2
    grating_pitch   = average(pitch_min, pitch_max)
    grating_depth   = average(h1_min, h1_max)
    box1_width      = average(box_width_min, box_width_max)/2
    box2_width      = box1_width/2
    box_centre_dist = average(box_centre_dist_min, box_centre_dist_max)
    box1_eps        = average(box_eps_min, box_eps_max)
    box2_eps        = box1_eps
    gaussian_width  = average(gaussian_width_min, gaussian_width_max)
    substrate_depth = average(substrate_depth_min, substrate_depth_max)
    substrate_eps   = average(substrate_eps_min, substrate_eps_max)

def Initial_bigrating():
    return grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, gaussian_width_value, substrate_depth, substrate_eps

