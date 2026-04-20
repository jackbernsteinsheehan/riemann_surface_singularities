import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tqdm import tqdm

# Save the state at every moment in time, and then plot at end
# caluclate a geodesic at each moment, see how it changes with the curvature
# Jost 2.3 up to curvature?? Fubiini-Study metric

# beta in [0, 1]
BETA = 0.5

def sim_in_polar(a=1.0, t=1.0, Nr=80, Ntheta=80):

    # Get radii in [0,1]
    r = np.linspace(0.0, 1.0, Nr)
    # Get theta from [-pi, pi]
    theta = np.linspace(-np.pi, np.pi, Ntheta, endpoint=False)

    # Calculate radial and angular steps
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]
    
    #FIXME will update if we extend to non-radially symmetric
    # Stable dt

  
    dt = (dr**2) / (4*a) / 2
    t_nodes = int(t / dt) + 1

    # Initialize u, where u[t, i, j] is the temp at time t, radius i, angle j
    u = np.zeros((t_nodes, Nr, Ntheta), dtype=float)


    # Set init condition
    u[0, :, :] = 1

    # Set boundary condition
    u[0] = set_boundary(u[0])

    # The model: updates for each time step t
    for n in tqdm(range(t_nodes - 1)):
        w = u[n]
        #print(np.log(w))
        log_w = np.log(w)


        # Update excludes r = 0
        for i in range(1, Nr - 1):
            radius = r[i]
            #Second derivative
            u_rr = (log_w[i+1, :] - 2*log_w[i, :] + log_w[i-1, :]) / dr ** 2

            # First derivative
            u_r = (log_w[i+1, :] - log_w[i-1, :]) / (2 * dr)

            # Apply to next slice
            u[n+1, i, :] =  u[n, i, :] + dt * a * radius**(2-(2*BETA)) * (u_rr + (1/radius * u_r))

        # Expensive...can we update in place?
        u[n+1] = set_boundary(u[n+1])

        # set r = 0 to the temp of the closest point
        u[n+1, 0, :] = u[n+1, 1, 0]


    R, TH = np.meshgrid(r, theta, indexing="ij")

    phi = TH
    X = R * np.cos(phi)
    Y = R * np.sin(phi)

    # Plot the sim
    plot_u_as_animation(u, X, Y, dt)


def set_boundary(w:np.ndarray):
    '''returns a copy of w with boundary conditions enforced. W should be a 2d array representing
    the temp at one time t. w[i, j] is temp at radius i, angle j for a given time.'''
    l = w.copy()
    l[-1, :] = 2
    return l


def plot_u_as_animation(u, X, Y, dt, interval_ms=100):
    t_nodes = u.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.18)

    zmin = float(u.min())
    zmax = float(u.max())
    ax.set_zlim(zmin, zmax)

    k0 = 0
    surf = ax.plot_surface(X, Y, u[k0], cmap="jet", vmin=zmin, vmax=zmax, shade=True)
    ax.set_title(f"t = {k0*dt:.4f} s (k={k0})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Temp")

    current_frame = {"idx": 0}
    timer = fig.canvas.new_timer(interval=interval_ms)

    def draw_frame(k):
        nonlocal surf
        surf.remove()
        surf = ax.plot_surface(X, Y, u[k], cmap="jet", vmin=zmin, vmax=zmax, shade=True)
        ax.set_title(f"t = {k*dt:.4f} s (k={k})")
        return (surf,)

    def step():
        next_idx = (current_frame["idx"] + 1) % t_nodes
        current_frame["idx"] = next_idx
        draw_frame(next_idx)
        fig.canvas.draw_idle()

    timer.add_callback(step)

    rewind_ax = fig.add_axes([0.23, 0.05, 0.16, 0.05])
    pause_ax = fig.add_axes([0.42, 0.05, 0.12, 0.05])
    play_ax = fig.add_axes([0.57, 0.05, 0.12, 0.05])
    rewind_button = Button(rewind_ax, "Rewind")
    pause_button = Button(pause_ax, "Pause")
    play_button = Button(play_ax, "Play")

    def rewind(_event):
        timer.stop()
        current_frame["idx"] = 0
        draw_frame(0)
        fig.canvas.draw_idle()

    def pause(_event):
        timer.stop()

    def play(_event):
        timer.start()

    rewind_button.on_clicked(rewind)
    pause_button.on_clicked(pause)
    play_button.on_clicked(play)
    fig._timer = timer
    plt.show()


def initialize(u, dx, dy):
    Nt, Nx, Ny = u.shape

    # Get coordinate arrays
    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dy

    # f(0, y) = 0
    u[:, 0, :] = 0

    # f(1, y) = 1-2y
    u[:, -1, :] = 1 - (2 * y)

    # f(x, 0) = sin((pi * x) /2)
    u[:, :, 0] = np.sin(((np.pi / 2) * x))

    # f(x, 1) = -sin((pi * x) /2)
    u[:, :, -1] = -np.sin(((np.pi / 2) * x))

    return u


if __name__ == "__main__":
    sim_in_polar(t=1)
