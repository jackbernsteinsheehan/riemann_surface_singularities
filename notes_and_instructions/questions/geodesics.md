Are we studying the trajectory of a geodesic over one weight function at a specific time, or updating the trajectory at the same time that we update the weight function? i.e., should I calculate gamma_dot_dot and update the position and velocity, and then move to the next weight function a time dt later and calculate gamma_dot_dot again based on the new curvature, plotting the geodesic trajectory as the metric flows? After writing all this I'm thinking this is probably the goal.

I'm using the `frames` object from the `sim_in_polar` function from `curvature_flow.py` as representative of the weight function for simulating geodesics. Do we want to go more fine-grained than this? Additionally, is the Nr=30 and Ntheta=20 that we've been using go to be specific enough to calculate trajectories?

**Answers:**

The first step would be to calculate the whole trajectory for one single weight function, preferably in a function like `geodesic(initial_position, initial_velocity, weight_function)`.  The output would be a 1D array of complex numbers which we can overlay on a plot of u itself.  Then, we can apply the function to each frame throughout the curvature flow.

As for Nr and Ntheta, I have no idea how much granularity we need.  Once we have it working we can see how large those numbers must be for the path to appear to converge.