import argparse
import numpy as np
import matplotlib.pyplot as plt


def fubini_study_geodesic(z0, dz0, output, t_max=4, points=1000):
    """
    Plot the Fubini–Study geodesic determined by
    initial point z0 and initial tangent dz0.
    """

    fig, (ax_z, ax_w) = plt.subplots(1, 2, figsize=(12, 6))

    theta = np.linspace(0, 2*np.pi, 200)
    for ax in [ax_z, ax_w]:
        ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1, alpha=0.4)
        ax.set_aspect('equal')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, linestyle=':', alpha=0.6)

    ax_z.set_title(r"$z$-chart ($|z|\leq1$)")
    ax_w.set_title(r"$w=1/z$-chart ($|w|\leq1$)")

    # --- stereographic projection to sphere ---
    def to_sphere(z):
        r2 = abs(z)**2
        return np.array([
            2*z.real,
            2*z.imag,
            1-r2
        ])/(1+r2)

    def to_complex(p):
        x,y,z = p
        return (x + 1j*y)/(1+z)

    # lift point
    p0 = to_sphere(z0)

    # compute pushforward of tangent
    # derivative of stereographic projection
    scale = 2/(1+abs(z0)**2)
    v0 = np.array([
        scale*dz0.real,
        scale*dz0.imag,
        -2*scale*(z0.real*dz0.real + z0.imag*dz0.imag)
    ])

    # orthogonalize
    v0 -= np.dot(v0,p0)*p0
    v0 /= np.linalg.norm(v0)

    # generate great circle
    t = np.linspace(-t_max, t_max, points)
    sphere_pts = np.cos(t)[:,None]*p0 + np.sin(t)[:,None]*v0

    # project back
    z_vals = np.array([to_complex(p) for p in sphere_pts])

    # split into charts
    mask_z = abs(z_vals) <= 1
    mask_w = abs(z_vals) >= 1

    ax_z.plot(z_vals[mask_z].real, z_vals[mask_z].imag, 'r', lw=2)
    w_vals = 1/z_vals[mask_w]
    ax_w.plot(w_vals.real, w_vals.imag, 'b', lw=2)

    # mark initial point
    if abs(z0)<=1:
        ax_z.scatter(z0.real, z0.imag, color='green', zorder=5)
    else:
        w0 = 1/z0
        ax_w.scatter(w0.real, w0.imag, color='green', zorder=5)

    plt.tight_layout()
    if output:
        plt.savefig(output)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot Fubini–Study geodesic")
    parser.add_argument("--point", type=complex, default=0.2+0.8j, help="Initial point z0 in the complex plane")
    parser.add_argument("--tangent", type=complex, default=0.5+0.1j, help="Initial tangent dz0 as a complex number")
    parser.add_argument("--output", default=None, help="File name to save geodesic graph.")

    args = parser.parse_args()

    fubini_study_geodesic(args.point, args.tangent, args.output)

if __name__ == "__main__":
    main()
