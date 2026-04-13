import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tqdm import tqdm
import config

# Jost 2.3 up to curvature?? Fubiini-Study metric

BETA = 0

def sim_in_polar(a=1.0, t=1.0, Nr=30, Ntheta=90):

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
    u_next = np.zeros_like(u)

    # The model: updates for each time step t
    for n in tqdm(range(t_nodes - 1)):
        w = u
        u_next[:, :] = w[:, :]

        u_next[1:-1] = w[1:-1] + dt * a* laplacian(w,r,dr,theta,dtheta)

        #OLD VERSION
        # Update excludes r = 0
        # for i in range(1, Nr - 1):
        #     radius = r[i]
        #     for j in range(Ntheta):
            
        #         # Wrap around when we get to the beginning or end.
        #         # The neighbor should be the opposite index

        #         jp = (j + 1) % Ntheta
        #         jm = (j - 1) % Ntheta

        #         #Second derivative of r
        #         u_rr = (w[i+1, j] - 2*w[i, j] + w[i-1, j]) / dr ** 2

        #         # First derivative of r
        #         u_r = (w[i+1, j] - w[i-1, j]) / (2 * dr)

        #         # Second derivative of theta
        #         u_theta_theta = (w[i, jp] - 2*w[i, j] + w[i, jm])/ dtheta**2

        #         # Apply to next slice
        #         u_next[i, j] = w[i, j] + dt * a * (u_rr + (1/radius * u_r) + 1/(radius**2)*(u_theta_theta))

        # Update in place instead of calling the function...might be faster
        u_next[-1, :] = np.cos(6 * theta)
        

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
    plot_frames_as_animation(frames, frame_times, X, Y)

#Calculate laplacian
def laplacian(f,r,dr,theta,dtheta):
    f_rr = (f[2:] - 2 * f[1:-1] + f[:-2]) / (dr ** 2)
    f_r = (f[2:] - f[:-2]) / (2 * dr)
    f_theta_theta = (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / (dtheta ** 2)

    return f_rr + f_r / r[1:-1, np.newaxis] + f_theta_theta[1:-1] / (r[1:-1, np.newaxis] ** 2)

    # First derivative of r
    u_r = (w[i+1, j] - w[i-1, j]) / (2 * dr)

    # Second derivative of theta
    u_theta_theta = (w[i, jp] - 2*w[i, j] + w[i, jm])/ dtheta**2

def set_boundary(w:np.ndarray, theta):
    '''returns a copy of w with boundary conditions enforced. W should be a 2d array representing
    the temp at one time t. w[i, j] is temp at radius i, angle j for a given time.'''
    l = w.copy()
    l[-1, :] = np.cos(6 * theta)
    return l


def plot_frames_as_animation(frames, frame_times, X, Y, interval_ms=100):
    if len(frames) == 0:
        raise ValueError("frames is empty")

    if frame_times is None or len(frame_times) != len(frames):
        frame_times = [float(k) for k in range(len(frames))]

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

    current_frame = {"idx": 0}
    timer = fig.canvas.new_timer(interval=interval_ms)

    def draw_frame(k):
        nonlocal surf
        Z = frames[k].copy()
        Z[0, 1:] = np.nan
        surf.remove()
        surf = ax.plot_surface(X, Y, Z, cmap="jet", vmin=zmin, vmax=zmax, shade=True)
        ax.set_title(f"t = {frame_times[k]:.4f} s (frame={k})")
        return (surf,)

    def step():
        next_idx = (current_frame["idx"] + 1) % len(frames)
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

if __name__ == "__main__":
    sim_in_polar()
