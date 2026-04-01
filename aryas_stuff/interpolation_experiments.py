import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib
import matplotlib.pyplot
import matplotlib.colors
import matplotlib.cm
import matplotlib.widgets
from tqdm import tqdm

# 1. Define coarse input grid
r = np.linspace(0, 10, 5)
theta = np.linspace(0, 2*np.pi, 6, endpoint=True)
R, TH = np.meshgrid(r, theta, indexing='ij')
Z = R**2 * np.cos(4*TH)  # Sample data

# 2. Create the interpolator (default is cubic)
# Note: Z should be transposed if created with meshgrid(indexing='xy')
interp_func = RegularGridInterpolator((r, theta), Z, method='linear')

print(interp_func([9,1]))

# TO DO
# Plot the rough data
# Produce path in r, theta coords
# Add path to plot of surface