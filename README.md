# Ilic-and-optimisation

This code enables one to predict the beam-riding stability of a Gaussian-irradiated bigrating, and dynamically evolve the relatvistic equations of motion. Stability is predicted by the eigenvalues $\xi$ formed from the Jacobian matrix of phase space $\vec{x}=[y,\phi,v_y,\dot{\phi}]$ within frame $\mathcal{U}$, an instantaneous, upwards comoving frame. The Jacobian terms are given by: 

```math
\displaylines{k^y_y =- D^2 \frac{I_0}{mc} \left[Q^{R'}_{2} - Q^{L'}_{2} \right](0, \lambda') \left\{1 - \exp\left[-\frac{1}{2 \bar{w}^2} \right] \right\} \\
    k^y_\phi =-D^2 \frac{I_0}{mc} \left[ \frac{\partial Q^{R'}_{2}}{\partial \delta'} + \frac{\partial Q^{L'}_{2}}{\partial \delta'} \right](0, \lambda') \frac{w}{2} \sqrt{\frac{\pi}{2}} \text{erf}\left[\frac{1}{\bar{w} \sqrt{2}}  \right] \\
    \mu^y_y=-D^2 \frac{I_0}{mc} \frac{1}{c} \frac{D+1}{D(\gamma+1)} \left[ Q^{R'}_{\text{pr,1}} + Q^{L'}_{\text{pr,1}} + \frac{\partial Q^{R'}_{\text{pr,2}}}{\partial \delta'} + \frac{\partial Q^{L'}_{\text{pr,2}}}{\partial \delta'} \right](0, \lambda') \, \frac{w}{2} \sqrt{\frac{\pi}{2}} \text{erf}\left[\frac{1}{\bar{w} \sqrt{2}} \right],  \\
    \mu^y_\phi =D^2 \frac{I_0}{mc} \frac{1}{c} \left(2 \left[Q^{R'}_{2} - Q^{L'}_{2}\right] - D \left[ \frac{\partial  Q^{R'}_{2}}{\partial \bar{\nu}'} - \frac{\partial  Q^{L'}_{2}}{\partial \bar{\nu}'} \right]\right) (0, \lambda') \frac{w^2}{4} \left(1 - \exp\left[-\frac{1}{2\bar{w}^2} \right] \right)}
```

with the Jacobian matrix given as 
```math
J=\begin{bmatrix}
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1 \\
    k^y_y & k^y_\phi & \mu^y_y & \mu^y_\phi \\
    k^\phi_y & k^\phi_\phi & \mu^\phi_y & \mu^\phi_\phi 
\end{bmatrix}.
```

# Folder structure

## `/Grating_and_parameters`
This folder contains the central class ``TwoBox`` in ``twobox.py`` which builds the grating using GRCWA[^1], and has a number of class definitions to evaluate the optical response (``self.eff()``, ``self.Q()``, ``return_Qs_auto()``) and predict stability (``self.Eigs()``). Special relativistic functions necessary for dynamics are stored in ``SR_functions.py``, while user-defined parameters are in ``parameters.py``. 

## ``/Linear stability``
Within this folder, ``Stability_diagram.py`` calls ``TwoBox.Eigs()`` over a user-defined veloctity and gaussian-width range, enabling a stability diagram to be produced in ``Stability_diagram.ipynb``. The Jupyter notebook ``Grating_stability.ipynb`` produces plots displaying a grating's Jacobian-matrix elements $J_{ij}$, eigenvalues $\xi$ and efficiency factors $Q_{pr,j'}'$.

## ``/Dynamics``
Since calculation of the efficiency factors is numerically time-intensive, "lookup tables" are formed prior to numerical integration.

### ``/Lookup_tables``
- `Lookup_tables.py`: calculates the efficiency factors $Q_{pr,j'}'(\delta',\lambda')$ and their derivatives $\partial / \partial \delta', \partial / \partial \lambda'$ over a user-defined angular and wavelength range, as necessary for linear interpolation in ``/Dynamics/Integrators``.
- `Lookup_tables.py`: performs similar to `Lookup_tables.py`, but only over either a given angular range, or wavelength range.

### ``/Integrators``
This folder contains three dynamics integrators to evolve lightsail dynamics:
- `!M_non-linear.py`: this is the main method. The full non-linear equations of motion are applied in frame $\mathcal{M}$, which is continually updating by converting between $\mathcal{M}$ and $\mathcal{L}$ as necessary.
- `M_non-dispersive.py`: this integrator only considers the grating response at $\lambda'=\lambda'(v=0)$, and updates information between $\mathcal{M}$ and $\mathcal{L}$ the same as `!M_non-linear.py`.
- `U_Jacobian-terms.py`: the linearised equations of motion are obtained from the derived Jacobian terms, which only being valid in frame $\mathcal{U}$, are hence applied in $\mathcal{U}$, requiring information to be updated between $\mathcal{U}$ and $\mathcal{L}$.

## ``/Optimisation``
This folder contains preliminary code for optimising a bigrating-Gaussian structure for asymptotic stability, where `opt.py` defines the global and local optimisation which is run by `run_parallel.py`, with results extracted in `run_parallel_extract.ipynb`.

[^1]: https://github.com/weiliangjinca/grcwa
