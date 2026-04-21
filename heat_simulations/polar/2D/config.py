# --- Curvature flow params --- #
A = 1
T = 1
Nr = 25
Ntheta = 25

# can be True, False, "single", "animate"
CURVE_PLOT = "animate" # Show curvature flow plot, a single frame with "single", or autoplay with "animate"
SAVE_EVERY = 100 # How often we save the state to a new frame
BETA = 0
RHO = 0



# --- Geodesic params --- #
P = 0.75 - 0.6j
V = -0.35 + 0.0j
U_IDX = 50 # determines which weight function from the frames object will be used
STEPS = 8000 # Number of iterations in geodesic calculation
GEO_PLOT = False # Show geodesic path on a 2D circle
