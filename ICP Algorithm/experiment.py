import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import function as ICP


# Initialize the parameters.
# Threshold is the limitation of acceptable error in match function
# Iteration is the iterating times
fig = plt.figure()
threshold = 20
iteration = 100
i = 0
file = "ICP_data/" + str(i) + ".ply"
frame0 = ICP.readfile(file)
plt.scatter(frame0.x, frame0.y, s=2, alpha=0.5)

frame_list = [frame0]

for i in range(9):
    i = i + 1
    print("正在处理第%d帧" %i)
    file = "ICP_data/" + str(i) + ".ply"
    frame = ICP.readfile(file)
    x1 = np.mat(frame.x)
    y1 = np.mat(frame.y)

    x0 = np.mat(frame_list[-1].x)
    y0 = np.mat(frame_list[-1].y)

    R, t, pos_new = ICP.transform(x0, y0, x1, y1, threshold)
    frame.x = pos_new[0].tolist()
    frame.y = pos_new[1].tolist()
    frame.move(R, t)

    plt.scatter(frame.x, frame.y, s=2, alpha=0.5)

    frame_list.append(frame)

ax = plt.gca()
ax.set_aspect(1)
plt.show()

sio.savemat('data.mat', {'frame_list': frame_list})


