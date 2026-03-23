
# Heat Diffusion and Geodesics
We're modeling heat diffusion and investigating curvature. This process began with simple simulations of heat diffusion on a 1D rod, and then a 2D surface. Then we began varying the boundary conditions, and transitioned to simulations in polar coordinates using the polar form for the laplacian.

Our long-term goal is to model curvature flow on Riemann surfaces with prescribed conical singularities, producing the uniformization metrics whose existence was establish in *Ricci flow on surfaces with conic singularities* (2015) by Mazzeo, Rubinstein, Sesum.

### Installation
To run the simulations, you'll need to install `python`, and run
```powershell
pip3 install numpy
```
```powershell
pip3 install matplotlib
```
to install the necessary stack.

## `heat_simulations`
### `heat_simulations/first_passes`
In `first_passes`, you'll find the first simulations we created, like the initial 1D and 2D models, as well as experiments with more interesting boundary conditions. With `bounds_time_stack` we made our first attempt at storing all values and plotting all at once.

### `heat_simulations/polar`
In `polar`, you'll find our two directories:
### 2D:
In the `polar/2D` directory, we have the `curvature_flow.py`, `curvature_flow_on_sphere.py` and the `polar_2d.py` modules. 
`curvature_flow.py` is a simulation of (unnormalized) Ricci flow using the PDE $u_t=\Delta\log u$. Notice that the `sim_in_polar()` function takes default parameters for `a` (the diffusivity constant), `t` (the time interval over which we simulate), as well as the number of radial and angular samples. At the bottom of the file, you can tweak the parameters as you wish. Run the file to see the simulation. Here is an example: 
![alt text](img/conformal_cone.png)

`polar_2d.py` is a simulation of classical heat flow, done in polar coordinates. You can similarly adjust the paramters as you wish. Here is a cool example from at two different time points: 
![alt text](img/polar_early_t.png)
![alt text](img/polar_late_t.png)

`curvature_flow_on_sphere.py` is a simulation of Yamabe flow on a sphere. This is a work in progress, and is not yet fully implemented. Here is an example of the glueing from south and north pole: ![alt text](image.png)

### symmetric:
In the `polar/symmetric` folder are our first simulations in polar coordinates, which use radially symmetric initial conditions, allowing us to omit the angular derivative component of the laplacian. Here's an example of the one simulating curvature over a disk: ![alt text](img/cone.png)

## `geodesics`
In the `geodesics` directory, you can see some attempts at modeling the trajectory of a geodesic near a cone structure. The animations are rudimentary (very ugly!).


## Log
Working on geodesics. Need to use r, theta to apply the index of the frames object to infer the actual radii and anglular values

