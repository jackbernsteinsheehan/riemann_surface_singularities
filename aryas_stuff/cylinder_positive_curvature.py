import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
#import trig
import tqdm

# beta in [0, 1]
BETA = 1

# rho is the target scalar curvature
RHO = .5

yMIN = -1
yMAX = 1.0
tMAX = 5.0
a = 5.0
ySAMPLES = 40
thetaSAMPLES = 11

yVals = np.linspace(yMIN, yMAX, ySAMPLES)

# Get theta from [-pi, pi]
thetaVals = np.linspace(-np.pi, np.pi, thetaSAMPLES, endpoint=True)

# Calculate radial and angular steps
dy = yVals[1] - yVals[0]
dtheta = thetaVals[1] - thetaVals[0]

# Stable dt for no diffusion dependent on theta
dt = (dy**2) / (4*a) / 2
tSamples = int(tMAX / dt) + 1

# Initial metric coefficient -- 1 gives rise to flat conical metric
def ff(x):
    return 1

# Background metric
muBackground = (ff(yVals) * np.exp((BETA - 1) * yVals))[:,np.newaxis] * np.full((thetaSAMPLES),1)

# Calculates laplacian.  Output loses both y endpoints.
# This version uses broadcasting to perform the operation in C, which is about twice as fast as the old version.
def standard_laplacian(f):
    f_yy = (f[2:] - 2 * f[1:-1] + f[:-2]) / (dy ** 2)
    return np.exp(-2 * yVals[1:-1])[:, np.newaxis] * f_yy

# Calculates laplacian with respect to a metric coefficient mu
def metric_laplacian(f, mu):
    return -standard_laplacian(f) / (mu[1:-1] ** 2)

# Calculates curvature
def measure_curvature(mu):
    return metric_laplacian(np.log(mu), mu)

# Integrates a function
def integrate(f, mu):
    return np.sum(mu[1:-1] * np.exp(2 * BETA * yVals[1:-1])[:, np.newaxis] * f * dy * dtheta)

# First order extrapolation on the r=0 end
def left_extrapolate(f):
    firstRow = f[0] + (f[0] - f[1])
    return np.concatenate((firstRow[None,:], f), axis=0)

# Measures average curvature
def average_curvature(mu):
    curvature = measure_curvature(mu)
    return integrate(standard_laplacian(np.log(mu)), np.ones_like(mu)) / integrate(np.ones_like(curvature), mu)

# Converts u data into isometric embedding in R^3.  Only works for radially symmetric u
def isometricEmbedding(u):
    cVals = np.zeros((u.shape[0] + 1, u.shape[1]))
    hVals = np.zeros((u.shape[0] + 1, u.shape[1]))
    cVals[1:] =  np.sqrt(u) * np.exp(yVals[:, np.newaxis])
    dc = cVals[1:, 0] - cVals[:-1, 0]
    dd = dy * np.sqrt(u[:, 0]) * np.exp(yVals)
    dh = np.sqrt(np.abs(dd ** 2 - dc ** 2))
    dh[0] = dh[0] / 4
    hVals[1:, :] = np.cumsum(dh)[:, np.newaxis]
    return np.array([cVals, hVals])

def sim_in_polar(t=tMAX):
    # Background scalar curvature
    KBackground = measure_curvature(muBackground)

    # Initialize u, where u[t, i, j] is the temp at time t, radius i, angle j
    u = np.ones((tSamples, ySAMPLES, thetaSAMPLES), dtype=float)

    avgCurvOverTime=np.array([])

    # Set init condition
    u[0, :, :] = 1.0
    u[:, -1, :] = 1.0


    # New version
    lam = u * muBackground
    gMetric = lam * np.exp(- yVals)[:, np.newaxis]
    print(average_curvature(lam[0]))

    for n in tqdm.tqdm(range(tSamples-1)):
        avgCurvOverTime=np.append(avgCurvOverTime,average_curvature(gMetric[n]))

        # Update excludes y endpoints.
        #lam[n+1, 1:-1] = lam[n, 1:-1] + dt * (average_curvature(gMetric[n]) - measure_curvature(gMetric[n])) * lam[n,1:-1]
        lam[n+1, 1:-1] = lam[n, 1:-1] + dt * (-RHO - measure_curvature(lam[n])) * lam[n,1:-1]

        # First order extrapolation to update left endpoint
        lam[n+1, 0] = lam[n+1, 1]# + (lam[n+1, 1] - lam[n+1, 2])

        gMetric[n+1] = lam[n+1] * np.exp(2 * yVals)[:, np.newaxis]

    avgCurvOverTime = np.append(avgCurvOverTime, average_curvature(gMetric[-1]))

    print(measure_curvature(lam[-1]))

    #print(isometricEmbedding(lam[-1]))

    Y, TH = np.meshgrid(yVals, thetaVals, indexing="ij")

    iso_frames = [isometricEmbedding(lam[i]) for i in range(tSamples)]

    # Plot the sim
    plot_side_by_side_with_slider(lam, gMetric, iso_frames, Y, TH, dt, avgCurvOverTime)


def plot_side_by_side_with_slider(u1, u2, iso_frames, Y_mesh, TH_mesh, dt, avgCurv, title1="u", title2="g"):
    tSamples = u1.shape[0]
    fig = plt.figure(figsize=(12, 10))
    
    # Subplot 1
    ax1 = fig.add_subplot(221, projection="3d")
    zmin1 = float(u1.min())
    zmax1 = float(u1.max())
    ax1.set_zlim(zmin1, zmax1)
    ax1.set_xlabel("y")
    ax1.set_ylabel("theta")
    ax1.set_zlabel(title1)

    # Subplot 2
    ax2 = fig.add_subplot(222, projection="3d")
    zmin2 = float(u2.min())
    zmax2 = float(u2.max())
    ax2.set_zlim(zmin2, zmax2)
    ax2.set_xlabel("y")
    ax2.set_ylabel("theta")
    ax2.set_zlabel(title2)

    # Subplot 3
    ax3 = fig.add_subplot(223)
    ax3.plot(np.arange(len(avgCurv)) * dt, avgCurv)
    ax3.set_xlabel("t")
    ax3.set_ylabel("Average Curvature")
    ax3.set_title("Average Curvature over Time")
    
    # Subplot 4
    iso_ax = fig.add_subplot(224, projection="3d")
    
    all_iso_h = []
    all_iso_c = []
    for cs, hs in iso_frames:
        if not np.isnan(hs).all():
            all_iso_h.extend([np.nanmin(hs), np.nanmax(hs)])
        if not np.isnan(cs).all():
            all_iso_c.append(np.nanmax(cs))
            
    iso_hmin = float(min(all_iso_h)) if all_iso_h else 0.0
    iso_hmax = float(max(all_iso_h)) if all_iso_h else 1.0
    iso_cmax = float(max(all_iso_c)) if all_iso_c else 1.0
    iso_ax.set_zlim(iso_hmin, iso_hmax)
    iso_ax.set_xlim(-iso_cmax, iso_cmax)
    iso_ax.set_ylim(-iso_cmax, iso_cmax)
    iso_ax.set_xlabel("x")
    iso_ax.set_ylabel("y")
    iso_ax.set_zlabel("height")

    k0 = 0
    vline = ax3.axvline(k0 * dt, color='r')

    fig.subplots_adjust(bottom=0.18)

    surfaces = {}

    def draw_frame(k):
        # Remove old surfaces
        for surface in surfaces.values():
            surface.remove()
        surfaces.clear()

        surfaces["u1"] = ax1.plot_surface(Y_mesh, TH_mesh, u1[k], cmap="jet", vmin=zmin1, vmax=zmax1, shade=True)
        ax1.set_title(f"{title1} at t = {k*dt:.4f} s")
        
        surfaces["u2"] = ax2.plot_surface(Y_mesh, TH_mesh, u2[k], cmap="jet", vmin=zmin2, vmax=zmax2, shade=True)
        ax2.set_title(f"{title2} at t = {k*dt:.4f} s")
        
        c_vals, h_vals = iso_frames[k]
        X_iso = c_vals * np.cos(TH_mesh[0])
        Y_iso = c_vals * np.sin(TH_mesh[0])
        Z_iso = h_vals
        
        surfaces["iso"] = iso_ax.plot_surface(X_iso, Y_iso, Z_iso, cmap="jet", shade=True, edgecolor='k', linewidth=0.1)
        iso_ax.set_title(f"Isometric Embedding at t = {k*dt:.4f} s")
        
        vline.set_xdata([k*dt, k*dt])

    draw_frame(k0)

    slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.04])
    s = Slider(slider_ax, "time index k", 0, tSamples - 1, valinit=k0, valstep=1)

    def update(val):
        k = int(s.val)
        draw_frame(k)
        fig.canvas.draw_idle()

    s.on_changed(update)
    plt.show()


if __name__ == "__main__":
    sim_in_polar()