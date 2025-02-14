"""
Calculates just one slice (1D array) of 2D (lambda, delta) array from `Lookup_tables.py'
Two options:
- Fixed wavelength - varying angle
- Fixed angle (zero) - varying wavelength
"""

import sys
sys.path.append('../')

import numpy as np
import pickle
from Grating_and_parameters.twobox import TwoBox
from Grating_and_parameters.parameters import D1_ND, gaussian_width
import time


t_start = time.time()

grating_type = "Ilic"

klambda = 1000
kdelta = 1000

####################
## Build grating

if grating_type == "Ilic":
    ## Data for Ilic, w=2 (up to 10%c)
    wavelength              = 1.5 
    grating_pitch           = 1.8 / wavelength
    grating_depth           = 0.5 / wavelength
    box1_width              = 0.15 * grating_pitch
    box2_width              = 0.35 * grating_pitch
    box_centre_dist         = 0.60 * grating_pitch
    box1_eps                = 3.5**2 
    box2_eps                = 3.5**2
    gaussian_width_value    = gaussian_width(grating_type) # not needed here but oh well
    substrate_depth         = 0.5 / wavelength
    substrate_eps           = 1.45**2

    ## Final velocity, hence wavelength
    v_final = 10/100
    grating_type_print = "Ilic-10"

if grating_type == "Ilic-damp":
    ## Data for stable Ilic, w=2.718
    wavelength              = 1.5 / D1_ND(1.2/100)
    grating_pitch           = 1.8 / wavelength
    grating_depth           = 0.5 / wavelength
    box1_width              = 0.15 * grating_pitch
    box2_width              = 0.35 * grating_pitch
    box_centre_dist         = 0.60 * grating_pitch
    box1_eps                = 3.5**2 
    box2_eps                = 3.5**2
    gaussian_width_value    = gaussian_width(grating_type) # not needed here but oh well
    substrate_depth         = 0.5 / wavelength
    substrate_eps           = 1.45**2

    ## Final velocity, hence wavelength
    v_final = 6.8/100
    grating_type_print = grating_type

wavelength      = 1.
angle           = 0.    # delta=0 if varying wavelength
Nx              = 100
numG            = 25
Qabs            = np.inf

grating = TwoBox(grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, 
                 gaussian_width, substrate_depth, substrate_eps,
                 wavelength, angle, Nx, numG, Qabs)

####################
## Number of lambda' points
lambda_final = 1/D1_ND(v_final)
lambda_array = np.linspace( wavelength, lambda_final, klambda )

## Number of delta' points
delta_max = 15 * (np.pi / 180)
delta_min = - delta_max
delta_array  = np.linspace( delta_min, delta_max, kdelta )

####################
Option = "vary-wavelength"
## Pick case
if Option == "vary-wavelength":
    k = klambda
    k_array = lambda_array
if Option == "vary-angle":
    k = kdelta
    k_array = delta_array

####################
## Storage arrays
Q1_array            = np.zeros( k )
Q2_array            = np.zeros( k )
PD_Q1_delta_array   = np.zeros( k )
PD_Q2_delta_array   = np.zeros( k )
PD_Q1_lambda_array  = np.zeros( k )
PD_Q2_lambda_array  = np.zeros( k )

for i in range(k):
    if Option == "vary-wavelength":
         grating.wavelength = k_array[i]
    if Option == "vary-angle":
        grating.angle   = k_array[i]
    
    # Call function
    Qs = grating.return_Qs_auto(return_Q=True)

    # Efficiency factors & derivatives
    Q1_array[i] = Qs[0];             Q2_array[i] = Qs[1]
    PD_Q1_delta_array[i] = Qs[2];   PD_Q2_delta_array[i] = Qs[3]
    PD_Q1_lambda_array[i] = Qs[4];  PD_Q2_lambda_array[i] = Qs[5]

t_end = time.time()-t_start
t_end_sec = round(t_end)
t_end_min = round(t_end/60)
t_end_hours = round(t_end/60**2)
print(rf"Finished in {t_end_sec} seconds, or {t_end_min} minutes, or {t_end_hours} hours!")

## Save data
pkl_fname = rf".{grating_type_print}_Lookup_table_k{k}_{Option}.pkl"
data = {'Q1': Q1_array,                         'Q2': Q2_array, 
        'PD_Q1_delta':PD_Q1_delta_array,        'PD_Q2_delta': PD_Q2_delta_array,
        'PD_Q1_lambda': PD_Q1_lambda_array,     'PD_Q2_lambda': PD_Q2_lambda_array,
        'k_array': k_array}
with open(pkl_fname, 'wb') as data_file:
    pickle.dump(data, data_file)
