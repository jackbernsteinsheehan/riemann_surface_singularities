import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm



BETA = 0
RHO = 0

def sim_in_polar(a=1.0, t=1, Nr=20, Ntheta=20):

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
    frames_south = []
    frames_north = []
    frames_iso_south = []
    frames_iso_north = []
    frames_k_south = []
    frames_k_north = []
    frame_times = []
    save_every = 20
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
            frames_south.append(u_south.copy())
            frames_north.append(u_north.copy())
            frames_iso_south.append(isometricPlot(u_south, r, Nr, dr, theta, dtheta))
            frames_iso_north.append(isometricPlot(u_north, r, Nr, dr, theta, dtheta))

            cur_R_s = laplacian(np.log(u_south), r, dr, theta, dtheta)
            K_s = np.zeros_like(u_south)
            K_s[1:-1] = cur_R_s / u_south[1:-1]
            K_s[0] = K_s[1].mean()
            K_s[-1] = K_s[-2]
            frames_k_south.append(K_s)
            
            cur_R_n = laplacian(np.log(u_north), r, dr, theta, dtheta)
            K_n = np.zeros_like(u_north)
            K_n[1:-1] = cur_R_n / u_north[1:-1]
            K_n[0] = K_n[1].mean()
            K_n[-1] = K_n[-2]
            frames_k_north.append(K_n)
            
            frame_times.append(n*dt)

    print(-laplacian(np.log(u_south),r,dr,theta,dtheta)/u_south[1:-1])

    R, TH = np.meshgrid(r, theta, indexing="ij")

    phi = (1-BETA) * TH
    X = R * np.cos(phi)
    Y = R * np.sin(phi)

    # Plot the sim
    plot_simulation(frames_south, frames_north, frames_iso_south, frames_iso_north, frames_k_south, frames_k_north, frame_times, X, Y, TH)

# Converts u data into isometric embedding in R^3.  Only works for radially symmetric u
def isometricPlot(u, r, Nr, dr, theta, dtheta):
    cVals = np.zeros_like(u)
    hVals = np.zeros_like(u)
    for i in range(Nr-1):
        cVals[i+1] = r[i+1, np.newaxis] * u[i+1] ** (1/2)
        dc = cVals[i+1, 0] - cVals[i, 0]
        dd = dr * u[i+1, 0] ** (1/2)
        hVals[i+1] = hVals[i] + np.abs(dd ** 2 - dc ** 2)**(1/2)
    return np.array([cVals, hVals])

# Cycles a list
def cycle_left(list:np.ndarray):
    newlist = list.copy()
    newlist = np.append(newlist, newlist[0])
    return newlist[1:]

def cycle_right(list:np.ndarray):
    newlist = list.copy()
    newlist = np.insert(newlist, 0, newlist[0])
    return newlist[:-1]

# Calculate laplacian
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


def plot_simulation(south_frames, north_frames, iso_south_frames, iso_north_frames, k_south_frames, k_north_frames, times, X, Y, TH):
    if not south_frames or not north_frames:
        raise ValueError("Frames lists cannot be empty.")

    if times is None or len(times) != len(south_frames):
        times = [float(k) for k in range(len(south_frames))]

    num_frames = len(south_frames)

    fig = plt.figure(figsize=(12, 10))
    fig.subplots_adjust(bottom=0.18)

    # Setup subplots and limits
    south_ax = fig.add_subplot(221, projection="3d")
    south_zmin = float(min(f.min() for f in south_frames))
    south_zmax = float(max(f.max() for f in south_frames))
    south_ax.set(zlim=(south_zmin, south_zmax), xlabel="x", ylabel="y", zlabel="u_south")

    north_ax = fig.add_subplot(222, projection="3d")
    north_zmin = float(min(f.min() for f in north_frames))
    north_zmax = float(max(f.max() for f in north_frames))
    north_ax.set(zlim=(north_zmin, north_zmax), xlabel="x", ylabel="y", zlabel="u_north")

    iso_ax = fig.add_subplot(212, projection="3d")

    all_k = [k for k_frames in (k_south_frames, k_north_frames) for k in k_frames]
    k_min = float(min(k.min() for k in all_k))
    k_max = float(max(k.max() for k in all_k))
    
    k_margin = (k_max - k_min) * 0.2 if k_max > k_min else 0.1
    norm = plt.Normalize(vmin=k_min, vmax=k_max + k_margin)
    
    all_iso_h = []
    all_iso_c = []
    for (cs, hs), (cn, hn) in zip(iso_south_frames, iso_north_frames):
        if not np.isnan(hs).all():
            all_iso_h.extend([np.nanmin(hs), np.nanmax(hs)])
        if not np.isnan(cs).all():
            all_iso_c.append(np.nanmax(cs))
            
        hn_shifted = hs[-1, 0] + hn[-1, 0] - hn
        if not np.isnan(hn_shifted).all():
            all_iso_h.extend([np.nanmin(hn_shifted), np.nanmax(hn_shifted)])
        if not np.isnan(cn).all():
            all_iso_c.append(np.nanmax(cn))
            
    iso_hmin = float(min(all_iso_h)) if all_iso_h else 0.0
    iso_hmax = float(max(all_iso_h)) if all_iso_h else 1.0
    iso_cmax = float(max(all_iso_c)) if all_iso_c else 1.0
    iso_ax.set(zlim=(iso_hmin, iso_hmax), xlim=(-iso_cmax, iso_cmax), ylim=(-iso_cmax, iso_cmax),
               xlabel="x", ylabel="y", zlabel="height")

    # Store surface objects so we can remove them before drawing the next frame
    surfaces = {}

    def draw_frame(frame_idx):
        # Remove old surfaces
        if surfaces:
            surfaces["south"].remove()
            surfaces["north"].remove()
            surfaces["iso_south"].remove()
            surfaces["iso_north"].remove()

        time_str = f"t = {times[frame_idx]:.4f} s (frame={frame_idx})"

        # Draw South
        Z_south = south_frames[frame_idx].copy()
        surfaces["south"] = south_ax.plot_surface(X, Y, Z_south, cmap="jet", vmin=south_zmin, vmax=south_zmax, shade=True)
        south_ax.set_title(f"u_south at {time_str}")

        # Draw North
        Z_north = north_frames[frame_idx].copy()
        surfaces["north"] = north_ax.plot_surface(X, Y, Z_north, cmap="jet", vmin=north_zmin, vmax=north_zmax, shade=True)
        north_ax.set_title(f"u_north at {time_str}")

        # Draw Isometric embedding (South)
        c_vals_s, h_vals_s = iso_south_frames[frame_idx]
        X_iso_s = c_vals_s * np.cos(TH)
        Y_iso_s = c_vals_s * np.sin(TH)
        Z_iso_s = h_vals_s.copy()
        K_s = k_south_frames[frame_idx]
        K_face_s = (K_s[:-1, :-1] + K_s[1:, :-1] + K_s[:-1, 1:] + K_s[1:, 1:]) / 4
        colors_s = plt.cm.jet_r(norm(K_face_s))
        surfaces["iso_south"] = iso_ax.plot_surface(X_iso_s, Y_iso_s, Z_iso_s, facecolors=colors_s, shade=True, edgecolor='k', linewidth=0.2)

        # Draw Isometric embedding (North)
        c_vals_n, h_vals_n = iso_north_frames[frame_idx]
        X_iso_n = c_vals_n * np.cos(TH)
        Y_iso_n = c_vals_n * np.sin(TH)
        Z_iso_n = h_vals_s[-1, 0] + h_vals_n[-1, 0] - h_vals_n.copy()
        K_n = k_north_frames[frame_idx]
        K_face_n = (K_n[:-1, :-1] + K_n[1:, :-1] + K_n[:-1, 1:] + K_n[1:, 1:]) / 4
        colors_n = plt.cm.jet_r(norm(K_face_n))
        surfaces["iso_north"] = iso_ax.plot_surface(X_iso_n[::-1], Y_iso_n[::-1], Z_iso_n[::-1], facecolors=colors_n[::-1], shade=True, edgecolor='k', linewidth=0.2)
        
        iso_ax.set_title(f"Isometric embedding at {time_str}")

    # Draw the initial frame
    draw_frame(0)

    # Setup the slider
    slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.04])
    time_slider = Slider(slider_ax, "frame", 0, num_frames - 1, valinit=0, valstep=1)

    def update(val):
        draw_frame(int(time_slider.val))
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()

if __name__ == "__main__":
    sim_in_polar()