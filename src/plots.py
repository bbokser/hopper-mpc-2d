"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
import matplotlib.animation as animation
plt.style.use(['science', 'no-latex'])
plt.rcParams['lines.linewidth'] = 2
import matplotlib.ticker as plticker
import itertools
plt.rcParams['font.size'] = 16


def fplot(total, p_hist, f_hist, s_hist):

    fig, axs = plt.subplots(5, sharex="all")
    plt.xlabel("Timesteps")

    axs[0].plot(range(total), p_hist[:, 2], color='blue')
    axs[0].set_title('Base z position w.r.t. time')
    axs[0].set_ylabel("Z position (m)")

    axs[1].plot(range(total), f_hist[:, 0], color='blue')
    axs[1].set_title('Magnitude of X Output Force')
    axs[1].set_ylabel("Force, N")
    axs[2].plot(range(total), f_hist[:, 1], color='blue')
    axs[2].set_title('Magnitude of Y Output Force')
    axs[2].set_ylabel("Force, N")
    axs[3].plot(range(total), f_hist[:, 2], color='blue')
    axs[3].set_title('Magnitude of Z Output Force')
    axs[3].set_ylabel("Force, N")
    axs[4].plot(range(total), s_hist, color='blue')
    axs[4].set_title('Scheduled Contact')
    axs[4].set_ylabel("True/False")

    plt.show()


def posplot(p_ref, p_hist):

    ax = plt.axes(projection='3d')
    ax.plot(p_hist[:, 0], p_hist[:, 1], p_hist[:, 2], color='red', label='Body Position')
    ax.set_title('Body Position')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.scatter(*p_hist[0, :], color='green', marker="x", s=200, label='Starting Position')
    ax.scatter(*p_ref, marker="x", s=200, color='orange', label='Target Position')
    ax.legend()
    intervals = 2
    loc = plticker.MultipleLocator(base=intervals)
    ax.xaxis.set_minor_locator(loc)
    ax.yaxis.set_minor_locator(loc)
    ax.zaxis.set_minor_locator(loc)
    # Add the grid
    ax.grid(which='minor', axis='both', linestyle='-')
    ax.xaxis.labelpad = 30
    ax.yaxis.labelpad = 30
    ax.zaxis.labelpad = 30

    plt.show()


def posfplot(p_ref, p_hist, p_pred_hist, f_pred_hist, pf_hist):

    ax = plt.axes(projection='3d')
    ax.plot(p_hist[:, 0], p_hist[:, 1], p_hist[:, 2], color='red', label='Point CoM Pos')
    ax.set_title('Body Position')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.scatter(*p_hist[0, :], color='green', marker="x", s=200, label='Starting Position')
    ax.scatter(p_pred_hist[:, 0], p_pred_hist[:, 1], p_pred_hist[:, 2],
               color='purple', marker="o", s=200, label='MPC Predicted Positions')
    ax.quiver(p_pred_hist[:, 0], p_pred_hist[:, 1], p_pred_hist[:, 2],
              -f_pred_hist[:, 0], -f_pred_hist[:, 1], -f_pred_hist[:, 2], label='Predicted Forces')
    ax.scatter(pf_hist[:, 0], pf_hist[:, 1], pf_hist[:, 2], marker=".", s=200, color='blue', label='Footstep Pos')
    ax.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2], ls="--", color='g', label='Init Ref Traj')
    ax.legend()
    intervals = 2
    loc = plticker.MultipleLocator(base=intervals)
    ax.xaxis.set_minor_locator(loc)
    ax.yaxis.set_minor_locator(loc)
    ax.zaxis.set_minor_locator(loc)
    # Add the grid
    ax.grid(which='minor', axis='both', linestyle='-')
    ax.xaxis.labelpad = 30
    ax.yaxis.labelpad = 30
    ax.zaxis.labelpad = 30

    plt.show()


def animate_line(N, dataSet1, dataSet2, line, pf, ax):
    line._offsets3d = (dataSet1[0:3, :N])
    # ref._offsets3d = (dataSet2[0:3, :N])
    pf._offsets3d = (dataSet2[0:3, :N])
    # ax.view_init(elev=10., azim=N)


def posplot_animate(p_f, p_hist, ref_traj, p_pred_hist, f_pred_hist):  # , pf_ref):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title('Body Position')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim3d(0, 2)
    ax.set_ylim3d(0, 2)
    ax.set_zlim3d(0, 2)

    ax.scatter(*p_hist[0, :], color='green', marker="x", s=200, label='Starting Position')
    ax.scatter(*p_f, marker="x", s=200, color='orange', label='Target Position')
    ax.scatter(p_pred_hist[:, 0], p_pred_hist[:, 1], p_pred_hist[:, 2],
               color='purple', marker="o", s=200, label='MPC Predicted Positions')
    ax.quiver(p_pred_hist[:, 0], p_pred_hist[:, 1], p_pred_hist[:, 2],
              -f_pred_hist[:, 0], -f_pred_hist[:, 1], -f_pred_hist[:, 2], label='Predicted Forces')
    intervals = 2
    loc = plticker.MultipleLocator(base=intervals)
    ax.xaxis.set_minor_locator(loc)
    ax.yaxis.set_minor_locator(loc)
    ax.zaxis.set_minor_locator(loc)
    # Add the grid
    ax.grid(which='minor', axis='both', linestyle='-')
    ax.xaxis.labelpad = 30
    ax.yaxis.labelpad = 30
    ax.zaxis.labelpad = 30

    N = len(p_hist)
    line = ax.scatter(p_hist[:, 0], p_hist[:, 1], p_hist[:, 2], lw=2, c='r', label='CoM Position')  # For line plot
    ref = ax.scatter(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], lw=2, c='g', label='Reference Trajectory')
    # pf = ax.scatter(pf_ref[:, 0], pf_ref[:, 1], pf_ref[:, 2], color='blue', label='Planned Footsteps')
    ax.legend()
    line_ani = animation.FuncAnimation(fig, animate_line, frames=N,
                                       fargs=(p_hist.T, ref_traj.T, line, ref, ax), # fargs=(p_hist.T, ref_traj.T, pf_ref.T, line, ref, pf, ax),
                                       interval=2, blit=False)
    # line_ani.save('basic_animation.mp4', fps=30, bitrate=4000, extra_args=['-vcodec', 'libx264'])

    plt.show()
