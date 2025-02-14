"""
For a given grating, calculate all the necessary data to 
"""
import sys
sys.path.append("../")
from Grating_and_parameters.SR_functions import Parameters, erf
from Grating_and_parameters.twobox import TwoBox
from Grating_and_parameters.parameters import D1_ND
import numpy as np
from numpy import linalg as LA
import pickle, time

I, L, m, c = Parameters()
runID = 1

#################
## Parameters
grating_type = "Ilic" # or 'Ilic_damp
final_speed_percent  = 15
final_speed = final_speed_percent/100

k_width = 1000
width_start = 0.01
width_end = 5

k_lambda = 4000
v_start = 0
v_end = final_speed

## Bound Arrays
width_array = np.linspace(width_start, width_end, k_width)
v_array = np.linspace(v_start, v_end, k_lambda)
lambda_array = np.zeros(k_lambda)
for i in range(k_lambda):
    lambda_array[i] = 1 / D1_ND(v_array[i])

#################
## Saving data arrays

# Jacobian term matrices
ky_y_array=       np.zeros((k_lambda, k_width))
ky_phi_array=     np.zeros((k_lambda, k_width))
muy_y_array=      np.zeros((k_lambda, k_width))
muy_phi_array=    np.zeros((k_lambda, k_width))
kphi_y_array=     np.zeros((k_lambda, k_width))
kphi_phi_array=   np.zeros((k_lambda, k_width))
muphi_y_array=    np.zeros((k_lambda, k_width))
muphi_phi_array=  np.zeros((k_lambda, k_width))

# Eigenvalue matrices
# R   =   np.zeros((k_lambda, k_width))
# Imag=   np.zeros((k_lambda, k_width))
neg_real    = np.zeros((k_lambda, k_width))
zero_real   = np.zeros((k_lambda, k_width))
neg_imag    = np.zeros((k_lambda, k_width))
zero_imag   = np.zeros((k_lambda, k_width))
eig1        =0j*np.zeros((k_lambda, k_width))
eig2    =0j*np.zeros((k_lambda, k_width))
eig3    =0j*np.zeros((k_lambda, k_width))
eig4    =0j*np.zeros((k_lambda, k_width))

# Eigenvector matrics
vec1 = np.empty((k_lambda, k_width), dtype=object)
for i in range(k_lambda):
    for j in range(k_width):
        vec1[i, j] = 0j*np.zeros(4)
vec2=vec1
vec3=vec1
vec4=vec1

# Conditions matrix
Cond1=       np.zeros((k_lambda, k_width))
Cond2=       np.zeros((k_lambda, k_width))
Cond3=       np.zeros((k_lambda, k_width))
Cond4=       np.zeros((k_lambda, k_width))

#################
## Build grating

if grating_type=="Ilic":
    ## Ilic - normalise parameters to wavelength = 1
    wavelength      = 1.5 
    grating_pitch   = 1.8 / wavelength
    grating_depth   = 0.5 / wavelength
    box1_width      = 0.15 * grating_pitch
    box2_width      = 0.35 * grating_pitch
    box_centre_dist = 0.60 * grating_pitch
    box1_eps        = 3.5**2 
    box2_eps        = 3.5**2
    gaussian_width  = 2 * L
    substrate_depth = 0.5 / wavelength
    substrate_eps   = 1.45**2
if grating_type=="Optimised":
    ## Previously 'optimised' grating
    grating_pitch   = 1.5384469388251338
    grating_depth   = 0.5580762361523982
    box1_width      = 0.10227122552871484
    box2_width      = 0.07605954942866577
    box_centre_dist = 0.2669020979549422
    box1_eps        = 9.614975107945112
    box2_eps        = 9.382304398409568
    gaussian_width  = 33.916288616522735
    substrate_depth = 0.17299998450776535
    substrate_eps   = 9.423032644325023

wavelength = 1.
angle = 0.
Nx = 100
numG = 25
Qabs = np.inf

grating = TwoBox(grating_pitch, grating_depth, box1_width, box2_width, box_centre_dist, box1_eps, box2_eps, 
                 gaussian_width, substrate_depth, substrate_eps,
                 wavelength, angle, Nx, numG, Qabs)

#################
## Calculations

timeSTART=time.time()
## At a given wavelength calculate width-independent terms
for i in range(k_lambda):
    lam = lambda_array[i]

    # Retrieve efficiency factors
    grating.wavelength = lam
    Q1R, Q2R, dQ1ddeltaR, dQ2ddeltaR, dQ1dlambdaR, dQ2dlambdaR = grating.return_Qs_auto(return_Q=True)

    ## Convert velocity dependence to wavelength dependence
    D = 1/lam 
    g = (np.power(lam,2) + 1)/(2*lam) 

    ## Symmetry
    Q1L = Q1R;   Q2L = -Q2R;   
    dQ1ddeltaL  = -dQ1ddeltaR;    dQ2ddeltaL  = dQ2ddeltaR
    dQ1dlambdaL = dQ1dlambdaR;    dQ2dlambdaL = -dQ2dlambdaR

    ## Calculate width-independent terms
    # y acceleration
    ky_y= -     D**2 * (I/(m*c)) *  ( Q2R - Q2L ) 
    ky_phi= -   D**2 * (I/(m*c)) * ( dQ2ddeltaR + dQ2ddeltaL )
    muy_y= -    D**2 * (I/(m*c)) * (1/c) * ( (D+1)/(D*(g+1)) ) * ( Q1R + Q1L  + dQ2ddeltaR + dQ2ddeltaL ) 
    muy_phi=    D**2 * (I/(m*c)) * (1/c) * ( 2*( Q2R - Q2L ) - lam*( dQ2dlambdaR - dQ2dlambdaL ) )

    # phi acceleration
    kphi_y=     D**2 * (12*I/( m*c*L**2)) * ( Q1R + Q1L )
    kphi_phi=   D**2 * (12*I/( m*c*L**2)) * ( dQ1ddeltaR - dQ1ddeltaL - ( Q2R - Q2L ) )
    muphi_y=    D**2 * (12*I/( m*c*L**2)) * (1/c) * ( (D+1)/(D*(g+1)) ) * ( dQ1ddeltaR - dQ1ddeltaL - ( Q2R - Q2L ) )
    muphi_phi= -D**2 * (12*I/( m*c*L**2)) * (1/c) * ( 2*( Q1R + Q1L ) - lam*( dQ1dlambdaR + dQ1dlambdaL ) )

    ## Go across widths
    for j in range(k_width):
        w_bar = width_array[j]
        w = w_bar * L
        ## Jacobian terms
        # y acceleration
        ky_y_w = ky_y *                      ( 1 - np.exp(-1/(2*w_bar**2) ) )
        ky_phi_w = ky_phi *                  (w/2) * np.sqrt( np.pi/2 ) * erf( 1/(w_bar*np.sqrt(2)) )
        muy_y_w = muy_y *                    (w/2) * np.sqrt( np.pi/2 ) * erf( 1/(w_bar*np.sqrt(2)) )
        muy_phi_w = muy_phi *     (w/2)**2 * ( 1 - np.exp( -1/(2*w_bar**2) ))
        # phi acceleration
        kphi_y_w = kphi_y *                  ( (w/2)*np.sqrt( np.pi/2 )  * erf( 1/(w_bar*np.sqrt(2)))  - (L/2)* np.exp( -1/(2*w_bar**2) ) ) 
        kphi_phi_w = kphi_phi *   (w/2)**2 * ( 1 - np.exp( -1/(2*w_bar**2) ))
        muphi_y_w = muphi_y *     (w/2)**2 * ( 1 - np.exp( -1/(2*w_bar**2) ))
        muphi_phi_w = muphi_phi * (w/2)**2 * ( (w/2)*np.sqrt( np.pi/2 )  * erf( 1/(w_bar*np.sqrt(2)))  - (L/2)* np.exp( -1/(2*w_bar**2) ) ) 

        ## Save to arrays
        ky_y_array[i,j]=        ky_y_w
        ky_phi_array[i,j]=      ky_phi_w
        muy_y_array[i,j]=       muy_y_w
        muy_phi_array[i,j]=     muy_phi_w
        kphi_y_array[i,j]=      kphi_y_w
        kphi_phi_array[i,j]=    kphi_phi_w
        muphi_y_array[i,j]=     muphi_y_w
        muphi_phi_array[i,j]=   muphi_phi_w

        ## Build jacobian matrix
        J00=ky_y_w;   J01=ky_phi_w;     J02=muy_y_w;    J03=muy_phi_w
        J10=kphi_y_w; J11=kphi_phi_w;   J12=muphi_y_w;  J13=muphi_phi_w
        J=np.array([[0,0,1,0],
                    [0,0,0,1],
                    [J00,J01,  J02,J03],
                    [J10,J11,  J12,J13]])
        
        ## Find the eigenvalues    
        EIGVALVEC=LA.eig(J)
        eig=EIGVALVEC[0]
        eig1[i,j]=eig[0]
        eig2[i,j]=eig[1]
        eig3[i,j]=eig[2]
        eig4[i,j]=eig[3]

        ## Take the real and imaginary part
        EIGreal=np.real(eig)
        EIGimag=np.imag(eig)

        ## Save eigenvectors
        eigvec=EIGVALVEC[1]
        vec1[i,j]=eigvec[:,0]
        vec2[i,j]=eigvec[:,1]
        vec3[i,j]=eigvec[:,2]
        vec4[i,j]=eigvec[:,3]

        # Conditions for stability
        a1= - ( J02 + J13 )
        a2= ( J02*J13 - ( J00 + J11 + J03*J12 ) )
        a3= J00 * J13 + J11*J02 - ( J01*J12 +  J10*J03 )
        a4= J00 * J11 - J01 * J10
        
        cond1=  a1
        cond2=  a1*a2 - a3
        cond3=  a3*cond2 -a1**2 * a4
        cond4=  a4

        Cond1[i,j]=cond1
        Cond2[i,j]=cond2
        Cond3[i,j]=cond3
        Cond4[i,j]=cond4

        # Real parts
        neg_real[i,j]   =   sum(n<0 for n in EIGreal)
        zero_real[i,j]  =   sum(n==0 for n in EIGreal)
        # Imaginary parts
        neg_imag[i,j]   =   sum(n<0 for n in EIGimag)
        zero_imag[i,j]  =   sum(n==0 for n in EIGimag)

        # Build matrices based on coniditions
        # if sum(n >= 0 for n in EIGreal)==0:
        #     R[i,j]=0.1  # real - all negative
        # elif sum(n >= 0 for n in EIGreal)==1:
        #     R[i,j]=0.2  # real - one positive (or zero)
        # elif sum(n >= 0 for n in EIGreal)==2:
        #     R[i,j]=0.3  # real - two positive (or zero), negative
        # elif sum(n >= 0 for n in EIGreal)==3:
        #     R[i,j]=0.4
        # elif sum(n >= 0 for n in EIGreal)==4:
        #     R[i,j]=0.5  # real - all positive (or zero)

        # if sum(n >= 0 for n in EIGimag)==0:
        #     Imag[i,j]=0.1   # imag - all negative
        # elif sum(n >= 0 for n in EIGimag)==1:
        #     Imag[i,j]=0.2   # imag - one positive (or zero)
        # elif sum(n >= 0 for n in EIGimag)==2:
        #     Imag[i,j]=0.3  # imag - two positive (or zero), negative
        # elif sum(n >= 0 for n in EIGimag)==3:
        #     Imag[i,j]=0.4  # imag - three positive (or zero), one negative 
        # elif sum(n >= 0 for n in EIGimag)==4:
        #     Imag[i,j]=0.5

timeDIFF = time.time()-timeSTART
t_end = timeDIFF
t_end_sec = round(t_end)
t_end_min = round(t_end/60)
t_end_hours = round(t_end/60**2)
print(rf"duration: {t_end_sec} (sec), {t_end_min} min")
print(rf"#lambda: {k_lambda}, #width: {k_width}")

#################
## Saving - split into three (~GB) files for 'ease'

# data1 = {'grating_type': grating_type, 'Nx': Nx, 'numG': numG, 'Qabs': Qabs, 'Intensity': I, 'duration (min)':t_end_min, 
#         'start speed': v_start, 'final speed': final_speed_percent, 'start width': width_start, 'final width': width_end,
#         'grating length': L,
#         'lambda_array': lambda_array, 'v_array': v_array, 'width_array': width_array,
#         'real_stability': R, 'imag_stability': Imag}
data1 = {'grating_type': grating_type, 'Nx': Nx, 'numG': numG, 'Qabs': Qabs, 'Intensity': I, 'duration (min)':t_end_min, 
        'start speed': v_start, 'final speed': final_speed_percent, 'start width': width_start, 'final width': width_end,
        'grating length': L,
        'lambda_array': lambda_array, 'v_array': v_array, 'width_array': width_array,
        'neg_real': neg_real, 'zero_real':zero_real,
        'neg_imag': neg_imag, 'zero_imag':zero_imag,}
data2 = {'eig1': eig1, 'eig2': eig2, 'eig3': eig3, 'eig4': eig4,
        'vec1': vec1, 'vec2': vec2, 'vec3': vec3, 'vec4': vec4,
        'kyy': ky_y_array, 'kyphi': ky_phi_array, 'muyy': muy_y_array, 'muyphi': muy_phi_array,
        'kphiy': kphi_y_array, 'kphiphi': kphi_phi_array, 'muphiy': muphi_y_array, 'muphiphi': muphi_phi_array, 
        'condition1': Cond1, 'condition2': Cond2, 'condition3': Cond3, 'condition4': Cond4
         }
data3 = {'kyy': ky_y_array, 'kyphi': ky_phi_array, 'muyy': muy_y_array, 'muyphi': muy_phi_array,
        'kphiy': kphi_y_array, 'kphiphi': kphi_phi_array, 'muphiy': muphi_y_array, 'muphiphi': muphi_phi_array, 
        'condition1': Cond1, 'condition2': Cond2, 'condition3': Cond3, 'condition4': Cond4
         }

pkl_fname1 = rf'./Data/{runID}/{grating_type}_Stability_Diagram_klambda{k_lambda}_by_kwidth{k_width}_num_neg_zero.pkl'
pkl_fname2 = rf'./Data/{runID}/{grating_type}_Stability_Diagram_klambda{k_lambda}_by_kwidth{k_width}_eigvec.pkl'
pkl_fname3 = rf'./Data/{runID}/{grating_type}_Stability_Diagram_klambda{k_lambda}_by_kwidth{k_width}_Jacobianterms.pkl'

with open(pkl_fname1, 'wb') as data_file:
    pickle.dump(data1, data_file)
with open(pkl_fname2, 'wb') as data_file:
    pickle.dump(data2, data_file)
with open(pkl_fname3, 'wb') as data_file:
    pickle.dump(data3, data_file)

