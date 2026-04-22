import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import config
import geodesic_calc as geo

GEODPLOT = True

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

    # Perturb initial condition (comment out these lines to start with flat disks glued along boundary)
    # u_south[0:5,:] = .5
    # u_south[10:,:] = 2
    # u_north[0:5,:] = .5
    # u_north[10:,:] = 2

    # Alt version
    # u_south[10:] = 2 * u_south[10:] - rValues[10:,np.newaxis]
    # u_north[10:] = 2 * u_north[10:] - rValues[10:,np.newaxis]

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

    if GEODPLOT:
        frames_geodesic_path = [geo.geodesic(u_north, rValues, thetaValues, config.STEPS)]
    else:
        frames_geodesic_path = [[]]
    
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
        u_south[-1] = 1 / (1 + dr)**4 * np.flip(0.8 * u_north[-3] + 0.2 * u_north[-4])
        u_north[-1] = 1 / (1 + dr)**4 * np.flip(0.8 * u_south[-3] + 0.2 * u_south[-4])

        # Make them agree at r=1
        u_north[-2] = np.flip(u_south[-2])

        # Set r = 0 to the average of the points on the smallest radius
        u_south[0, :] = u_south[1, :].mean()
        u_north[0, :] = u_north[1, :].mean()

        if (n + 1) % config.SAVE_EVERY == 0:
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

            if GEODPLOT:
                frames_geodesic_path.append(geo.geodesic(u_north, rValues, thetaValues, config.STEPS))
            else:
                frames_geodesic_path.append([])
            
            frame_times.append((n + 1) * dt)

    R, TH = np.meshgrid(rValues, thetaValues, indexing="ij")

    X = R * np.cos(TH)
    Y = R * np.sin(TH)

    # Plot the sim
    plot_simulation(frames_south, frames_north, frames_iso_south, frames_iso_north, frames_k_south, frames_k_north, frames_geodesic_path, frame_times, X, Y, TH, rValues, thetaValues, rho_t)

# Converts u data into isometric embedding in R^3.  Only works for radially symmetric u
def isometricPlot(u, rValues, dr):
    cVals = np.zeros_like(u)
    hVals = np.zeros_like(u)
    cVals[1:] = rValues[1:, np.newaxis] * u[1:] ** 0.5
    dc = cVals[1:, 0] - cVals[:-1, 0]
    dd = dr * u[1:, 0] ** 0.5
    dh = np.sqrt(np.abs(dd ** 2 - dc ** 2))
    hVals[1:, :] = np.cumsum(dh)[:, np.newaxis] / 1.68 #cheating to make it round
    return np.array([cVals, hVals])


def plot_simulation(south_frames, north_frames, iso_south_frames, iso_north_frames, k_south_frames, k_north_frames, geodesic_frames, times, X, Y, TH, rValues, thetaValues, rho_t):
    if not south_frames or not north_frames:
        raise ValueError("Frames lists cannot be empty.")

    if times is None or len(times) != len(south_frames):
        times = [float(k) for k in range(len(south_frames))]

    num_frames = len(south_frames)

    # Calculate global min/max for color scaling
    all_k = [k for k_frames in (k_south_frames, k_north_frames) for k in k_frames]
    k_min = float(min(k.min() for k in all_k))
    k_max = float(max(k.max() for k in all_k))
    
    max_dev = max(abs(k_max - rho_t), abs(k_min - rho_t)) / 2
    k_margin = max_dev * 0.1 if max_dev > 0 else 0.1
    cmin = k_min - k_margin
    cmax = rho_t / 4 + max_dev + k_margin

    # Calculate global min/max for z-axis scaling of weight functions
    south_zmin = min(0.0, float(min(f.min() for f in south_frames)))
    south_zmax = float(max(f.max() for f in south_frames))
    north_zmin = min(0.0, float(min(f.min() for f in north_frames)))
    north_zmax = float(max(f.max() for f in north_frames))

    # Calculate global min/max for isometric embedding dimensions to lock axis bounds
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

    # Create figure with 3 subplots: north, south, and a larger isometric plot spanning 2 rows
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "surface"}, {"type": "surface", "rowspan": 2}],
               [{"type": "surface"}, None]],
        subplot_titles=("Northern Hemisphere Weight", "Isometric Embedding", "Southern Hemisphere Weight"),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    # Helper function to extract and format data for a single animation frame
    def get_frame_data(frame_idx):
        Z_south = south_frames[frame_idx]
        Z_north = north_frames[frame_idx]

        # Southern isometric embedding data
        c_vals_s, h_vals_s = iso_south_frames[frame_idx]
        X_iso_s = c_vals_s * np.cos(TH)
        Y_iso_s = c_vals_s * np.sin(TH)
        Z_iso_s = h_vals_s
        K_s = k_south_frames[frame_idx]

        # Northern isometric embedding data (shifted and inverted to glue to South)
        c_vals_n, h_vals_n = iso_north_frames[frame_idx]
        X_iso_n = c_vals_n * np.cos(TH)
        Y_iso_n = c_vals_n * np.sin(TH)
        Z_iso_n = h_vals_s[-1, 0] + h_vals_n[-1, 0] - h_vals_n
        K_n = k_north_frames[frame_idx]

        path = np.asarray(geodesic_frames[frame_idx], dtype=complex)
        if GEODPLOT and len(path) > 0:
            r_path = np.abs(path)
            theta_path = np.angle(path)
            z_path = np.zeros_like(r_path)
            x_path = r_path * np.cos(theta_path)
            y_path = r_path * np.sin(theta_path)

            # Interpolate to find geodesic path on isometric embedding (Top/North)
            interp_c = RegularGridInterpolator((rValues, thetaValues), c_vals_n, method='cubic', bounds_error=False, fill_value=None)
            interp_h = RegularGridInterpolator((rValues, thetaValues), Z_iso_n, method='cubic', bounds_error=False, fill_value=None)
            points = np.column_stack((r_path, theta_path))
            c_path = interp_c(points)
            h_path = interp_h(points)
            x_iso_path = c_path * np.cos(theta_path)
            y_iso_path = c_path * np.sin(theta_path)
            z_iso_path = h_path
        else:
            x_path, y_path, z_path = [None], [None], [None]
            x_iso_path, y_iso_path, z_iso_path = [None], [None], [None]

        return Z_south, Z_north, x_path, y_path, z_path, x_iso_path, y_iso_path, z_iso_path, X_iso_s, Y_iso_s, Z_iso_s, K_s, X_iso_n, Y_iso_n, Z_iso_n, K_n

    Z_south, Z_north, x_path, y_path, z_path, x_iso_path, y_iso_path, z_iso_path, X_iso_s, Y_iso_s, Z_iso_s, K_s, X_iso_n, Y_iso_n, Z_iso_n, K_n = get_frame_data(0)

    # Add initial traces for the first frame
    # Trace 0: Northern Hemisphere Weight Function
    fig.add_trace(go.Surface(x=X, y=Y, z=Z_north, colorscale='Jet', cmin=north_zmin, cmax=north_zmax, showscale=False), row=1, col=1)
    # Trace 1: Southern Hemisphere Weight Function
    fig.add_trace(go.Surface(x=X, y=Y, z=Z_south, colorscale='Jet', cmin=south_zmin, cmax=south_zmax, showscale=False), row=2, col=1)
    
    # Trace 2: Geodesic path on Northern Hemisphere
    fig.add_trace(go.Scatter3d(x=x_path, y=y_path, z=z_path, mode='lines', line=dict(color='black', width=4), showlegend=False), row=1, col=1)
    
    # Traces 3 & 4: Start and end markers for geodesic
    if GEODPLOT and x_path[0] is not None:
        start_x, start_y, start_z = [x_path[0]], [y_path[0]], [z_path[0]]
        end_x, end_y, end_z = [x_path[-1]], [y_path[-1]], [z_path[-1]]
    else:
        start_x, start_y, start_z = [None], [None], [None]
        end_x, end_y, end_z = [None], [None], [None]
        
    fig.add_trace(go.Scatter3d(x=start_x, y=start_y, z=start_z, mode='markers', marker=dict(color='green', size=5), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=end_x, y=end_y, z=end_z, mode='markers', marker=dict(color='red', size=5), showlegend=False), row=1, col=1)

    # Trace 5: Isometric Embedding of Southern Hemisphere
    fig.add_trace(go.Surface(x=X_iso_s, y=Y_iso_s, z=Z_iso_s, surfacecolor=K_s, colorscale='Jet', cmin=cmin, cmax=cmax, colorbar=dict(title="Curvature")), row=1, col=2)
    # Trace 6: Isometric Embedding of Northern Hemisphere
    fig.add_trace(go.Surface(x=X_iso_n[::-1], y=Y_iso_n[::-1], z=Z_iso_n[::-1], surfacecolor=K_n[::-1], colorscale='Jet', cmin=cmin, cmax=cmax, showscale=False), row=1, col=2)

    # Trace 7: Geodesic path on Isometric Embedding
    fig.add_trace(go.Scatter3d(x=x_iso_path, y=y_iso_path, z=z_iso_path, mode='lines', line=dict(color='black', width=8), showlegend=False), row=1, col=2)

    # Traces 8 & 9: Start and end markers for geodesic on isometric plot
    if GEODPLOT and x_iso_path[0] is not None:
        iso_start_x, iso_start_y, iso_start_z = [x_iso_path[0]], [y_iso_path[0]], [z_iso_path[0]]
        iso_end_x, iso_end_y, iso_end_z = [x_iso_path[-1]], [y_iso_path[-1]], [z_iso_path[-1]]
    else:
        iso_start_x, iso_start_y, iso_start_z = [None], [None], [None]
        iso_end_x, iso_end_y, iso_end_z = [None], [None], [None]
        
    fig.add_trace(go.Scatter3d(x=iso_start_x, y=iso_start_y, z=iso_start_z, mode='markers', marker=dict(color='green', size=7), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=iso_end_x, y=iso_end_y, z=iso_end_z, mode='markers', marker=dict(color='red', size=7), showlegend=False), row=1, col=2)

    # Precompute animation frames
    frames = []
    for i in range(num_frames):
        Z_s, Z_n, xp, yp, zp, xip, yip, zip_path, Xis, Yis, Zis, Ks, Xin, Yin, Zin, Kn = get_frame_data(i)
        
        if GEODPLOT and xp[0] is not None:
            start_x, start_y, start_z = [xp[0]], [yp[0]], [zp[0]]
            end_x, end_y, end_z = [xp[-1]], [yp[-1]], [zp[-1]]
            iso_start_x, iso_start_y, iso_start_z = [xip[0]], [yip[0]], [zip_path[0]]
            iso_end_x, iso_end_y, iso_end_z = [xip[-1]], [yip[-1]], [zip_path[-1]]
        else:
            start_x, start_y, start_z = [None], [None], [None]
            end_x, end_y, end_z = [None], [None], [None]
            iso_start_x, iso_start_y, iso_start_z = [None], [None], [None]
            iso_end_y, iso_end_z = [None], [None]

        frame = go.Frame(
            data=[
                go.Surface(z=Z_n), # trace 0: North weight
                go.Surface(z=Z_s), # trace 1: South weight
                go.Scatter3d(x=xp, y=yp, z=zp), # trace 2: Geodesic line
                go.Scatter3d(x=start_x, y=start_y, z=start_z), # trace 3: Geodesic start
                go.Scatter3d(x=end_x, y=end_y, z=end_z), # trace 4: Geodesic end
                go.Surface(x=Xis, y=Yis, z=Zis, surfacecolor=Ks), # trace 5: South isometric
                go.Surface(x=Xin[::-1], y=Yin[::-1], z=Zin[::-1], surfacecolor=Kn[::-1]), # trace 6: North isometric
                go.Scatter3d(x=xip, y=yip, z=zip_path), # trace 7: iso geodesic line
                go.Scatter3d(x=iso_start_x, y=iso_start_y, z=iso_start_z), # trace 8: iso geodesic start
                go.Scatter3d(x=iso_end_x, y=iso_end_y, z=iso_end_z) # trace 9: iso geodesic end
            ],
            name=str(i),
            traces=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        )
        frames.append(frame)

    fig.frames = frames

    max_xy = float(np.max(np.abs(X)))

    # Update layout to set static axis ranges and animation controls
    fig.update_layout(
        title="Curvature Flow Simulation",
        height=900,
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="u_north",
            xaxis=dict(range=[-max_xy, max_xy], autorange=False),
            yaxis=dict(range=[-max_xy, max_xy], autorange=False),
            zaxis=dict(range=[north_zmin, north_zmax], autorange=False),
            aspectmode='cube'
        ),
        scene2=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="height",
            xaxis=dict(range=[-iso_cmax, iso_cmax], autorange=False),
            yaxis=dict(range=[-iso_cmax, iso_cmax], autorange=False),
            # zaxis=dict(range=[iso_hmin, iso_hmax], autorange=False),
            zaxis=dict(range=[0, 1.5], autorange=False),
            aspectmode='cube'
        ),
        scene3=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="u_south",
            xaxis=dict(range=[-max_xy, max_xy], autorange=False),
            yaxis=dict(range=[-max_xy, max_xy], autorange=False),
            zaxis=dict(range=[south_zmin, south_zmax], autorange=False),
            aspectmode='cube'
        ),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Frame: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [str(i)],
                        {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}
                    ],
                    "label": str(i),
                    "method": "animate"
                } for i in range(num_frames)
            ]
        }]
    )

    fig.show()

if __name__ == "__main__":
    sim_in_polar()