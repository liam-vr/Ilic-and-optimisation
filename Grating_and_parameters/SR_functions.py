"""
Stores all the special-relativistic functions necessary:
- Lorentz factor, Doppler factor
- Relativistic addition of velocities
- Four-vector Lorentz boost
- Abberration angle
- Wigner rotation angle
- Angle correction factors ABCS
- Wigner correction factor E
- + error function
"""
import autograd
from autograd import numpy as np
from autograd.scipy.special import erf as autograd_erf
import scipy

##########################################
## Velocities normalised by speed of light
def gamma_ND(v):
    """
    Calculate the Lorentz gamma factor with an input non-dimensionalised speed/velocity.

    Parameters
    ----------
    v     : ND speed or two/three-velocity or list of two/three-velocities

    Returns
    -------
    gamma : Lorentz gamma factor
    """
    if not isinstance(v,(list,np.ndarray)):
        v = [v]
    v = np.array(v)
    
    if any(isinstance(i, np.ndarray) for i in v):
        vnorm = np.linalg.norm(v,axis=1)
    else:
        vnorm = np.linalg.norm(v)
    
    gamma = 1/np.sqrt(1-np.power(vnorm,2))
    return gamma

def D1_ND(v):
    """
    Calculate the D_1 Doppler factor with an input non-dimensionalised velocity.

    Parameters
    ----------
    v  : ND two/three-velocity of the moving frame or list of two/three-velocities

    Returns
    -------
    D1 : Doppler factor
    """
    if not isinstance(v,(list,np.ndarray)):
        v = [v]
    v = np.array(v)
    
    if any(isinstance(i, np.ndarray) for i in v):
        vx = np.array([i[0] for i in v])
    else:
        vx = np.array(v[0])

    D1 = gamma_ND(v)*(1-vx)
    return D1
##########################################

##########################################
##  All velocities are in m/s (NOT normalised by c)
c = 299792458

def norm_squared(v):
    """
    ## Inputs
    v: [vx, vy] 2D array
    ## Output
    """
    return v[0]**2 + v[1]**2

def Gamma(v):
    """
    ## Inputs
    v: [vx, vy] 2D array
    ## Output
    Lorentz gamma factor gamma(v)
    """
    return (1 - norm_squared(v)/c**2)**(-1/2)

def Dv(v):
    """
    ## Inputs
    v: [vx, vy] 2D array
    ## Output
    Doppler factor gamma(v)(1- vx/c)
    """
    return Gamma(v)*(1 - v[0]/c)

def vadd(v,u):
    """
    ## Inputs
    v: [vx, vy] 2D array
    u: [vx, vy] 2D array
    ## Output
    v + u - SR addition of velocities
    """
    g = Gamma(v)
    return (1/(1+np.dot(v/c,u/c))) * ( v + u/g + ( (g/(g+1)) * np.dot(v/c,u/c) * v ) )

def Lorentz(v,z):
    """
    ## Inputs
    v: [vx, vy] 2D array
    ## Output
    returns ([t',x',y'])
    """
    t = z[0]; x = z[1]; y = z[2]
    g = Gamma(v)
    f = g**2/(g+1)
    
    t2 = g * ( t - (v[0]*x)/c**2 - (v[1]*y)/c**2 )
    x2 = -(g*v[0]*t) + (1+f*(v[0]/c)**2) * x + f*((v[0]/c)*(v[1]/c)) * y
    y2 = -(g*v[1]*t) + f*((v[0]/c)*(v[1]/c)) * x + (1+f*(v[1]/c)**2) * y

    return np.array([t2,x2,y2])

## Angle corrections

def SinCosTheta(v):
    """
    ## Inputs
    v: [vx, vy] 2D array
    ## Output
    sin(theta'), cos(theta') - relativistic aberration
    """
    D = Dv(v)
    g = Gamma(v)
    sin = (1/D)*( -g*v[1]/c + (g**2/(g+1))*(v[0]*v[1])/c**2 )
    cos = (1/D)*( -g*v[0]/c + 1 + (g**2/(g+1))*(v[0]/c)**2 )
    theta = np.arcsin(sin)
    return sin, cos, theta

def SinCosEpsilon(v,u):
    """
    ## Inputs
    v: [vx, vy] 2D array
    u: [ux, uy] 2D array
    ## Output
    sin(eps), cos(eps), eps - rotation angle from M to M+1
    """
    gv = Gamma(v)
    gu = Gamma(u)
    g = gv * gu * (1 + np.dot(v/c,u/c) )
    cross = ( (u[0]/c) * (v[1]/c) - (u[1]/c) * (v[0]/c) )
    sin = cross * (gv*gu*(1+g+gv+gu)) / ( (1+g)*(1+gv)*(1+gu) )
    cos = (1+g+gv+gu)**2 /( (1+g)*(1+gv)*(1+gu) ) - 1
    eps = np.arcsin(sin)
    return sin, cos, eps

def ABSC(v,phi):
    """
    ## Inputs
    v: [vx, vy] 2D array \n
    phi': grating angle
    ## Output
    A,B,S,C - sin(theta'), cos(theta'), S,C - linear corrections
    """
    sintheta = SinCosTheta(v)[0]
    costheta = SinCosTheta(v)[1]
    cos = np.cos(phi)
    sin = np.sin(phi)
    g = Gamma(v)
    D = Dv(v)
    bx = v[0] / c
    by = v[1] / c
    dot = (bx*cos +g*by*sin)
    dot2 = (by*cos +g*bx*sin)
    
    A = sintheta * cos/(g*D) - sin/D - (
        sintheta*dot) + dot2/(D*(g+1))- (
        g*by*dot/D ) + (
        ((g**2*(g+2))/(D*(g+1)**2))*bx*by*dot  )
    
    B = -costheta*dot + (2*bx*cos)/(D*(g+1)) - (
        (g*bx*dot)/D + (g*by**2 * cos)/(D**2 *(g+1)) ) + (
        ( (g**2 *(g+2))/(D*(g+1)**2) ) * bx**2 * dot )
    S = cos * A - sin * B
    C = sin * A + cos * B
    return A, B, S, C

def E_eps(v, phi):
    """
    ## Inputs
    v: [vx, vy] 2D array \n
    phi': grating angle
    ## Output
    $\mathcal(E)$ - epsilon linear correction
    """
    g = Gamma(v)
    return (g/(g+1)) * ( np.sin(phi)*v[0]/c - np.cos(phi)*v[1]/c )

def erf(x):
    return autograd_erf(x)

