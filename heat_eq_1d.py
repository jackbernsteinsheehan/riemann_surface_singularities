import numpy as np
import matplotlib.pyplot as plt

# defs
a = 110
length = 50 #mm
time = 3 #sec diffusivity
nodes = 35

dx = length / (nodes - 1)
dt = dx ** 2 / (2 * a)

t_nodes = int(time/dt)

# Init conditions
u = np.zeros(nodes) + 20 # plate initially at 20 degrees c


# boundary conditions
u[0] = 100
u[-1] = 100

# visualization
fig, ax = plt.subplots()
img = ax.imshow(u[np.newaxis, :], aspect="auto", vmin=0, vmax=100, cmap="jet")
plt.colorbar(img, ax=ax)

# simulation
counter = 0

for step in range(t_nodes):
    w = u.copy()

    for i in range(1, nodes - 1):
        u[i] = dt * a * (w[i - 1] - (2 * w[i]) + w[i + 1]) / dx ** 2 + w[i]

    u[0] = 100
    u[-1] = 100

    counter += dt

    print('t: {:.3f} [s], Average temp: {:.2f} Celcius'.format(counter, np.average(u)))
    img.set_data(u[np.newaxis, :])
    ax.set_title(f"t = {counter:.3f} s, avg = {u.mean():.2f} °C")

    plt.pause(0.01)
    
plt.ioff()
plt.show()