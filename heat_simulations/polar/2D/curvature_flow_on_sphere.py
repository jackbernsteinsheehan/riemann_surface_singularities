import numpy as np
import matplotlib
import matplotlib.pyplot
import matplotlib.colors
import matplotlib.cm
import matplotlib.widgets
from tqdm import tqdm


def sim_in_polar(a=1.0, t=1, Nr=15, Ntheta=40):

    # Get radii in [0,1]
    rValues = np.linspace(0.0, 1.0, Nr)
    rValues = np.append(rValues, 1+ 1 / Nr)
    Nr+=1

    # Get theta from [-pi, pi]
    thetaValues = np.linspace(-np.pi, np.pi, Ntheta, endpoint=True)

    # Calculate radial and angular steps
    dr = rValues[1] - rValues[0]
    dtheta = thetaValues[1] - thetaValues[0]
    
    # Stable dt for no diffusion dependent on theta
    #FIXME will update for non-radially symmetric case
    r_min = rValues[1]

    dt = 1 / (
        2*a*(
            1/(dr**2) + 1/((r_min**2)*(dtheta**2))
        )
    )

    dt *= 0.8
    t_nodes = int(t / dt) + 1

    # Initialize u
    u_south = np.zeros((Nr, Ntheta), dtype=float)
    u_north = np.zeros((Nr, Ntheta), dtype=float)

    # Set init condition
    u_south[:, :] = 1
    u_north[:, :] = 1

    # set frames for displaying (parallel lists)
    save_every = 20

    # Calculate laplacian
    def laplacian(f):
        f_rr = (f[2:] - 2 * f[1:-1] + f[:-2]) / (dr ** 2)
        f_r = (f[2:] - f[:-2]) / (2 * dr)
        f_theta_theta = (np.roll(f[1:-1], -1, axis=1) - 2 * f[1:-1] + np.roll(f[1:-1], 1, axis=1)) / (dtheta ** 2)
        return f_rr + f_r / rValues[1:-1, np.newaxis] + f_theta_theta / (rValues[1:-1, np.newaxis] ** 2)

    # Calculates the average value of f on the disk with respect to weight function. The size of f is Nr-2.
    def functionAverage(f, weight):
        return np.sum(f * rValues[1:-1, np.newaxis] * dr * dtheta) / np.sum(weight[1:-1] * rValues[1:-1, np.newaxis] * dr * dtheta)

    # Record initial conditions to frame before the loop starts
    R_south_init = laplacian(np.log(u_south))
    R_north_init = laplacian(np.log(u_north))

    frames_south=[u_south.copy()]
    frames_north=[u_north.copy()]
    frames_iso_south = [isometricPlot(u_south, rValues, dr)]
    frames_iso_north = [isometricPlot(u_north, rValues, dr)]

    K_s_init = np.zeros_like(u_south)
    K_s_init[1:-1] = R_south_init / u_south[1:-1]
    K_s_init[0] = K_s_init[1].mean()
    K_s_init[-1] = K_s_init[-2]
    frames_k_south = [K_s_init]
    
    K_n_init = np.zeros_like(u_north)
    K_n_init[1:-1] = R_north_init / u_north[1:-1]
    K_n_init[0] = K_n_init[1].mean()
    K_n_init[-1] = K_n_init[-2]
    frames_k_north = [K_n_init]
    
    frame_times = [0.0]

    rho_t = 0
    for n in tqdm(range(t_nodes - 1)):
        #Calculate curvature on each disk
        R_south = -laplacian(np.log(u_south))
        R_north = -laplacian(np.log(u_north))

        #Calculate average curvature for normalized Ricci flow
        rho_t = (functionAverage(R_south, u_south) + functionAverage(R_north, u_north)) / 2

        u_south[1:-1] = u_south[1:-1] + dt * a* (rho_t * u_south[1:-1] - R_south)
        u_north[1:-1] = u_north[1:-1] + dt * a* (rho_t * u_north[1:-1] - R_north)

        # Update boundary by pulling back u_north * |dz|^2 to update u_south and vice versa
        u_south[-1] = 1 / (1 + dr)**4 * np.flip(u_north[-3])
        u_north[-1] = 1 / (1 + dr)**4 * np.flip(u_south[-3])

        # Set r = 0 to the average of the points on the smallest radius
        u_south[0, :] = u_south[1, :].mean()
        u_north[0, :] = u_north[1, :].mean()

        if (n + 1) % save_every == 0:
            frames_south.append(u_south.copy())
            frames_north.append(u_north.copy())
            frames_iso_south.append(isometricPlot(u_south, rValues, dr))
            frames_iso_north.append(isometricPlot(u_north, rValues, dr))

            K_s = np.zeros_like(u_south)
            K_s[1:-1] = R_south / u_south[1:-1]
            K_s[0] = K_s[1].mean()
            K_s[-1] = K_s[-2]
            frames_k_south.append(K_s)
            
            K_n = np.zeros_like(u_north)
            K_n[1:-1] = R_north / u_north[1:-1]
            K_n[0] = K_n[1].mean()
            K_n[-1] = K_n[-2]
            frames_k_north.append(K_n)
            
            frame_times.append((n + 1) * dt)

    R, TH = np.meshgrid(rValues, thetaValues, indexing="ij")

    X = R * np.cos(TH)
    Y = R * np.sin(TH)

    # Plot the sim
    plot_simulation(frames_south, frames_north, frames_iso_south, frames_iso_north, frames_k_south, frames_k_north, frame_times, X, Y, TH, rho_t)

# Converts u data into isometric embedding in R^3.  Only works for radially symmetric u
def isometricPlot(u, rValues, dr):
    cVals = np.zeros_like(u)
    hVals = np.zeros_like(u)
    cVals[1:] = rValues[1:, np.newaxis] * u[1:] ** 0.5
    dc = cVals[1:, 0] - cVals[:-1, 0]
    dd = dr * u[1:, 0] ** 0.5
    dh = np.sqrt(np.abs(dd ** 2 - dc ** 2))
    hVals[1:, :] = np.cumsum(dh)[:, np.newaxis]
    return np.array([cVals, hVals])


def plot_simulation(south_frames, north_frames, iso_south_frames, iso_north_frames, k_south_frames, k_north_frames, times, X, Y, TH, rho_t):
    if not south_frames or not north_frames:
        raise ValueError("Frames lists cannot be empty.")

    if times is None or len(times) != len(south_frames):
        times = [float(k) for k in range(len(south_frames))]

    num_frames = len(south_frames)

    fig = matplotlib.pyplot.figure(figsize=(12, 10))
    fig.subplots_adjust(bottom=0.18)

    # Setup subplots and limits
    south_ax = fig.add_subplot(2, 2, 1, projection="3d")
    south_zmin = float(min(f.min() for f in south_frames))
    south_zmax = float(max(f.max() for f in south_frames))
    south_ax.set(zlim=(south_zmin, south_zmax), xlabel="x", ylabel="y", zlabel="u_south")

    north_ax = fig.add_subplot(2, 2, 3, projection="3d")
    north_zmin = float(min(f.min() for f in north_frames))
    north_zmax = float(max(f.max() for f in north_frames))
    north_ax.set(zlim=(north_zmin, north_zmax), xlabel="x", ylabel="y", zlabel="u_north")

    iso_ax = fig.add_subplot(2, 2, (2, 4), projection="3d")

    all_k = [k for k_frames in (k_south_frames, k_north_frames) for k in k_frames]
    k_min = float(min(k.min() for k in all_k))
    k_max = float(max(k.max() for k in all_k))
    
    max_dev = max(abs(k_max - rho_t), abs(k_min - rho_t)) / 2
    k_margin = max_dev * 0.1 if max_dev > 0 else 0.1
    norm = matplotlib.colors.Normalize(vmin=k_min - k_margin, vmax=rho_t / 4 + max_dev + k_margin)
    
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

    # Add a colorbar to the right of the isometric plot
    sm = matplotlib.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    cax = fig.add_axes((0.88, 0.15, 0.02, 0.3))
    fig.colorbar(sm, cax=cax, label="Curvature")

    # Store surface objects so we can remove them before drawing the next frame
    surfaces = {}

    def draw_frame(frame_idx):
        # Remove old surfaces
        for surface in surfaces.values():
            surface.remove()
        surfaces.clear()

        time_str = f"t = {times[frame_idx]:.4f} s (frame={frame_idx})"

        # Draw South
        Z_south = south_frames[frame_idx]
        surfaces["south"] = south_ax.plot_surface(X, Y, Z_south, cmap="jet", vmin=south_zmin, vmax=south_zmax, shade=True)
        south_ax.set_title(f"Southern Hemisphere Weight Function\n{time_str}")

        # Draw North
        Z_north = north_frames[frame_idx]
        surfaces["north"] = north_ax.plot_surface(X, Y, Z_north, cmap="jet", vmin=north_zmin, vmax=north_zmax, shade=True)
        north_ax.set_title(f"Northern Hemisphere Weight Function\n{time_str}")

        # Draw Isometric embedding (South)
        c_vals_s, h_vals_s = iso_south_frames[frame_idx]
        X_iso_s = c_vals_s * np.cos(TH)
        Y_iso_s = c_vals_s * np.sin(TH)
        Z_iso_s = h_vals_s
        K_s = k_south_frames[frame_idx]
        K_face_s = (K_s[:-1, :-1] + K_s[1:, :-1] + K_s[:-1, 1:] + K_s[1:, 1:]) / 4
        colors_s = matplotlib.colormaps['jet'](norm(K_face_s))
        surfaces["iso_south"] = iso_ax.plot_surface(X_iso_s, Y_iso_s, Z_iso_s, facecolors=colors_s, shade=True, edgecolor='k', linewidth=0.1)

        # Draw Isometric embedding (North)
        c_vals_n, h_vals_n = iso_north_frames[frame_idx]
        X_iso_n = c_vals_n * np.cos(TH)
        Y_iso_n = c_vals_n * np.sin(TH)
        Z_iso_n = h_vals_s[-1, 0] + h_vals_n[-1, 0] - h_vals_n
        K_n = k_north_frames[frame_idx]
        K_face_n = (K_n[:-1, :-1] + K_n[1:, :-1] + K_n[:-1, 1:] + K_n[1:, 1:]) / 4
        colors_n = matplotlib.colormaps['jet'](norm(K_face_n))
        surfaces["iso_north"] = iso_ax.plot_surface(X_iso_n[::-1], Y_iso_n[::-1], Z_iso_n[::-1], facecolors=colors_n[::-1], shade=True, edgecolor='k', linewidth=0.1)
        
        iso_ax.set_title(f"Isometric Embedding at {time_str}")

    # Draw the initial frame
    draw_frame(0)

    # Setup the slider
    slider_ax = fig.add_axes((0.15, 0.06, 0.7, 0.04))
    time_slider = matplotlib.widgets.Slider(slider_ax, "frame", 0, num_frames - 1, valinit=0, valstep=1)

    def update(val):
        draw_frame(int(time_slider.val))
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    matplotlib.pyplot.show()

if __name__ == "__main__":
    sim_in_polar()