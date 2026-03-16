import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm



BETA = 0
RHO = 0

def sim_in_polar(a=1.0, t=2, Nr=30, Ntheta=10):

    # Get radii in [0,1]
    r = np.linspace(0.0, 1.0, Nr)
    r = np.append(r, 1+ 1 / Nr)
    Nr+=1

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

    u_south = np.zeros((Nr, Ntheta), dtype=float)
    u_north = np.zeros((Nr, Ntheta), dtype=float)


    # Set init condition
    u_south[:, :] = 1
    u_north[:, :] = 1


    # # Set boundary condition
    # u = set_boundary(u, theta)

    # set frames for displaying (parallel lists)
    frames = []
    frame_times = []
    save_every = 200
    # u_south_next = np.zeros_like(u_south)+1
    # u_north_next = np.zeros_like(u_north)+1

    # The model: updates for each time step t
    # w = u
    # w = np.log(w)
    for n in tqdm(range(t_nodes - 1)):
        #Calculate curvature on each disk
        R_south = laplacian(np.log(u_south),r,dr,theta,dtheta)
        R_north = laplacian(np.log(u_north),r,dr,theta,dtheta)

        #Calculate average curvature for normalized Ricci flow
        rho_t = (functionAverage(R_south, u_south, r, Nr, dr, theta, dtheta) + functionAverage(R_north, u_north, r, Nr, dr, theta, dtheta)) / 2

        u_south[1:-1] = u_south[1:-1] + dt * a* (-rho_t * u_south[1:-1] + R_south)
        u_north[1:-1] = u_north[1:-1] + dt * a* (-rho_t * u_north[1:-1] + R_north)

        # Update in place instead of calling the function...might be faster
        #u_next[-1, :] = np.cos(2 * theta) + 2
        u_south[-1] = 1 / (1 + dr)**4 * np.flip(u_north[-3])
        u_north[-1] = 1 / (1 + dr)**4 * np.flip(u_south[-3])

        # set r = 0 to the average of the points on the smallest radius
        u_south[0, :] = u_south[1, :].mean()
        u_north[0, :] = u_north[1, :].mean()

        if n % save_every == 0:
            frames.append(u_south.copy())
            frame_times.append(n*dt)

    print(-laplacian(np.log(u_south),r,dr,theta,dtheta)/u_south[1:-1])
    
    R, TH = np.meshgrid(r, theta, indexing="ij")

    phi = (1-BETA) * TH
    X = R * np.cos(phi)
    Y = R * np.sin(phi)

    # Plot the sim
    plot_frames_with_slider(frames, frame_times, X, Y)

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
def laplacian(f, r, dr, theta, dtheta):
    f_rr = (f[2:] - 2 * f[1:-1] + f[:-2]) / (dr ** 2)
    f_r = (f[2:] - f[:-2]) / (2 * dr)
    f_theta_theta = (np.apply_along_axis(cycle_left, 1, f) - 2 * f + np.apply_along_axis(cycle_right, 1, f)) / (dtheta ** 2)

    return f_rr + f_r / r[1:-1, np.newaxis] + f_theta_theta[1:-1] / (r[1:-1, np.newaxis] ** 2)

# Integrates a function f against weight function
def integrate(f, weight, r, Nr, dr, theta, dtheta):
    return np.sum(
        np.array([weight[i+1] * f[i] * r[i+1] * dr * dtheta for i in range(0, Nr-2)])
        )

# Calculates the average value of f on the disk with respect to weight function.  The size of f is Nr-2.
def functionAverage(f, weight, r, Nr, dr, theta, dtheta):
    return integrate(f, np.full((Nr),1), r, Nr, dr, theta, dtheta) / integrate(np.full((Nr-2),1), weight, r, Nr, dr, theta, dtheta)


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
    sim_in_polar()