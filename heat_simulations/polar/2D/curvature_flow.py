import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm
import config


BETA = config.BETA
RHO = config.RHO

def sim_in_polar(a=config.A, t=config.T, Nr=config.Nr, Ntheta=config.Ntheta, plot = config.CURVE_PLOT):

    # Get radii in [0,1]
    r = np.linspace(0.0, 1.0, Nr)
    # Get theta from [-pi, pi]
    theta = np.linspace(-np.pi, np.pi, Ntheta, endpoint=True)

    # Calculate radial and angular steps
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]
    
    # Stable dt for no diffusion dependent on theta
    #FIXME will update for non-radially symmetric case
    r_min = r[1]

    dt = 1 / (
        2*a*(
            1/(dr**2) + 1/((r_min**2)*(dtheta**2))

        )
    )

    dt *= 0.8

    t_nodes = int(t / dt) + 1

    print("dt =", dt)
    print("t_nodes =", t_nodes)


    # Initialize u. We're not storing an array for every t anymore, dt is too small and
    # there's too many steps

    u = np.zeros((Nr, Ntheta), dtype=float)


    # Set init condition
    u[:, :] = 2    

    # Set boundary condition
    u = set_boundary(u, theta)

    # set frames for displaying (parallel lists)
    frames = []
    frame_times = []
    save_every = config.SAVE_EVERY
    u_next = np.zeros_like(u)+2

    # The model: updates for each time step t
    # w = u
    # w = np.log(w)
    for n in tqdm(range(t_nodes - 1)):
        w = u_next
        u_next[:, :] = w[:, :]

        u_next[1:-1] = w[1:-1] + dt * a* (RHO * w[1:-1] + laplacian(np.log(w),r,dr,theta,dtheta))

        u_next[-1, :] = np.cos(2 * theta) + 2
        

        u, u_next = u_next, u
        # set r = 0 to the average of the points on the smallest radius
        u[0, :] = u[1, :].mean()

        if n % save_every == 0:
            frames.append(u.copy())
            frame_times.append(n*dt)
    
    R, TH = np.meshgrid(r, theta, indexing="ij")

    phi = (1-BETA) * TH
    X = R * np.cos(phi)
    Y = R * np.sin(phi)

    # Plot the sim
    if plot == True:
        plot_frames_with_slider(frames, frame_times, X, Y)
    elif config.CURVE_PLOT == "single":
        plot_single_frame(frames, frame_times, X, Y, config.U_IDX)

    return (frames, frame_times, r, theta)

#Cycles a list
def cycle_left(list:np.ndarray):
    newlist = list.copy()
    newlist = np.append(newlist, newlist[0])
    return newlist[1:]

def cycle_right(list:np.ndarray):
    newlist = list.copy()
    newlist = np.insert(newlist, 0, newlist[0])
    return newlist[:-1]

#Calculate laplacian
def laplacian(f,r,dr,theta,dtheta):
    f_rr = (f[2:] - 2 * f[1:-1] + f[:-2]) / (dr ** 2)
    f_r = (f[2:] - f[:-2]) / (2 * dr)
    f_theta_theta = (np.apply_along_axis(cycle_left, 1, f) - 2 * f + np.apply_along_axis(cycle_right, 1, f)) / (dtheta ** 2)

    return f_rr + f_r / r[1:-1, np.newaxis] + f_theta_theta[1:-1] / (r[1:-1, np.newaxis] ** 2)


def set_boundary(w:np.ndarray, theta):
    '''returns a copy of w with boundary conditions enforced. W should be a 2d array representing
    the temp at one time t. w[i, j] is temp at radius i, angle j for a given time.'''
    l = w.copy()
    l[-1, :] = 2 + np.cos(10 * theta)
    return l


def plot_single_frame(frames, frame_times, X, Y, frame_idx=0):
    if len(frames) == 0:
        raise ValueError("frames is empty")

    if frame_times is None or len(frame_times) != len(frames):
        frame_times = [float(k) for k in range(len(frames))]

    if frame_idx < 0 or frame_idx >= len(frames):
        raise IndexError(f"frame_idx {frame_idx} is out of range for {len(frames)} saved frames")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.18)

    zmin = float(min(f.min() for f in frames))
    zmax = float(max(f.max() for f in frames))
    ax.set_zlim(zmin, zmax)

    Z = frames[frame_idx].copy()
    Z[0, 1:] = np.nan
    ax.plot_surface(X, Y, Z, cmap="jet", vmin=zmin, vmax=zmax, shade=True)

    ax.set_title(f"t = {frame_times[frame_idx]:.4f} s (frame={frame_idx})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Temp")

    plt.show()


def plot_frames_with_slider(frames, frame_times, X, Y):
    if len(frames) == 0:
        raise ValueError("frames is empty")

    if frame_times is None or len(frame_times) != len(frames):
        frame_times = [float(k) for k in range(len(frames))]

    n_frames = len(frames)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.18)

    zmin = float(min(f.min() for f in frames))
    zmax = float(max(f.max() for f in frames))
    ax.set_zlim(zmin, zmax)

    # ---- initial frame ----
    k0 = 0
    Z0 = frames[k0].copy()
    Z0[0, 1:] = np.nan          # mask center fan triangles (optional but recommended)
    surf = ax.plot_surface(X, Y, Z0, cmap="jet", vmin=zmin, vmax=zmax, shade=True)

    ax.set_title(f"t = {frame_times[k0]:.4f} s (frame={k0})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Temp")

    slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.04])
    s = Slider(slider_ax, "frame", 0, n_frames - 1, valinit=k0, valstep=1)

    def update(val):
        nonlocal surf
        k = int(s.val)          # <-- k is defined here

        Z = frames[k].copy()
        Z[0, 1:] = np.nan       # same masking on updates

        surf.remove()
        surf = ax.plot_surface(X, Y, Z, cmap="jet", vmin=zmin, vmax=zmax, shade=True)
        ax.set_title(f"t = {frame_times[k]:.4f} s (frame={k})")
        fig.canvas.draw_idle()

    s.on_changed(update)
    plt.show()

if __name__ == "__main__":
    sim_in_polar(config.A, config.T, config.Nr, config.Ntheta, config.CURVE_PLOT)
