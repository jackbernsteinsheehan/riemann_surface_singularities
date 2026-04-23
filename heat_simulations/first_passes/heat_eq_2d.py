import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import math as m

def plot_u_as_animation(u_frames, X, Y, times, interval_ms=100):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.18)

    zmin = 0
    zmax = 100
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("Temp (°C)")

    k0 = 0
    surf = ax.plot_surface(X, Y, u_frames[k0], cmap="jet", vmin=zmin, vmax=zmax, shade=True)
    ax.set_title(f"t = {times[k0]:.3f} s, avg = {u_frames[k0].mean():.2f} °C")

    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)

    current_frame = {"idx": 0}
    timer = fig.canvas.new_timer(interval=interval_ms)

    def draw_frame(k):
        nonlocal surf
        surf.remove()
        surf = ax.plot_surface(X, Y, u_frames[k], cmap="jet", vmin=zmin, vmax=zmax, shade=True)
        ax.set_title(f"t = {times[k]:.3f} s, avg = {u_frames[k].mean():.2f} °C")
        return (surf,)

    def step():
        next_idx = (current_frame["idx"] + 1) % len(u_frames)
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
    timer.start()
    plt.show()

def sim_u(a, length, width, time, nodes, BCx, BCy):

    # calculate additional parameters
    dx = length / (nodes - 1)
    dy = width / (nodes - 1)
    dt = min(dx ** 2 / (4 * a), dy ** 2 / (4 * a))

    t_nodes = int(time / dt)

    # Init conditions
    u = np.zeros((nodes, nodes)) + 40 # plate initially at 20 degrees c

    # boundary conditions
    # u[0, :] = BCx
    # u[-1, :] = BCy
    u[0:5,0:10] = 80
    u[20:30,0:2] = 15
    u[20:30,-2:-1] = 15
    u[18:21,18:21] = 90

    # visualization
    x = np.linspace(0, length, nodes)
    y = np.linspace(0, length, nodes)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # model
    counter = 0
    
    u_frames = [u.copy()]
    times = [0.0]
    save_every = max(1, t_nodes // 100) # Save about 100 frames to keep the slider smooth

    for step in range(t_nodes):
        w = u.copy()

        for i in range(1, nodes - 1):
            for j in range(1, nodes - 1):
                dd_ux = (w[i - 1, j] - 2*w[i, j] + w[i + 1, j]) / dx ** 2
                dd_uy = (w[i, j - 1] - 2*w[i, j] + w[i, j + 1]) / dy ** 2
            
                u[i, j] = dt * a * (dd_ux + dd_uy) + w[i, j]

        # u[0, :] = BCx
        # u[-1, :] = BCy

        # Neumann condition: derivative = 0
        u[0,:] = u[1,:]
        u[-1,:] = u[-2,:]
        u[:,0] = u[:,1]
        u[:,-1] = u[:,-2]

        # Reinforce boundary conditions
        u[0:5,0:10] = 80
        u[20:30,0:2] = 15
        u[20:30,-2:-1] = 15

        counter += dt

        if (step + 1) % save_every == 0 or step == t_nodes - 1:
            u_frames.append(u.copy())
            times.append(counter)
            #print('t: {:.3f} [s], Average temp: {:.2f} Celcius'.format(counter, np.average(u)))

    plot_u_as_animation(u_frames, X, Y, times)

# The first entry determines the speed of heat diffusion.  Use a=1 for slow, a=50 for fast

sim_u(50, 50, 35, 10, 40, 100, 100)