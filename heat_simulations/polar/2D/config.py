# --- Curvature flow params --- #
A = 1
T = 1
Nr = 25
Ntheta = 25

# can be True, False, "single"
CURVE_PLOT = False # Show curvature flow plot, or show a single frame with "single"
SAVE_EVERY = 200 # How often we save the state to a new frame
BETA = 0
RHO = 0



# --- Geodesic params --- #
P = 0.75 - 0.6j
V = -0.35 + 0.0j
U_IDX = 50 # determines which weight function from the frames object will be used
STEPS = 5000 # Number of iterations in geodesic calculation
GEO_PLOT = True # Show geodesic path on a 2D circle