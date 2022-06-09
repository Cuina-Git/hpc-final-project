import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

data = []
with open("data.log", "r") as file:
    line = file.readlines()
    for temp in line:
        temp = temp.strip('\n') 
        if temp.startswith('Process'):
            continue
        if temp.startswith('Vec'):
            continue
        if temp.startswith('  type'):
            continue
        else:
            data.append(float(temp))

x = np.linspace(0.0, 1.0, num=101)
t = np.linspace(0.0, 1.0, num=21)
x,t = np.array(np.meshgrid(x, t))
data = np.array(data)
data = np.reshape(data,(21,101))

# plot figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title("u(x,t)")
plt.xlabel("t(sec)")
plt.ylabel("x")
# Plot a basic wireframe.
ax.plot_wireframe(t, x, data, rstride=20, cstride=20)
plt.show()