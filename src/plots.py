"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import itertools


def fplot(total, p_hist, f_hist, s_hist):

    fig, axs = plt.subplots(5, sharex="all")
    plt.xlabel("Timesteps")

    axs[0].plot(range(total), p_hist[:, 1], color='blue')
    axs[0].set_title('Base z position w.r.t. time')
    axs[0].set_ylabel("Z position (m)")

    axs[1].plot(range(total), f_hist[:, 0], color='blue')
    axs[1].set_title('Magnitude of X Output Force')
    axs[1].set_ylabel("Force, N")

    axs[2].plot(range(total), f_hist[:, 1], color='blue')
    axs[2].set_title('Magnitude of Z Output Force')  # .set_title('angular velocity q1_dot')
    axs[2].set_ylabel("Force, N")  # .set_ylabel("angular velocity, rpm")

    axs[3].plot(range(total), s_hist[:, 0], color='blue')
    axs[3].set_title('Scheduled Contact')  # .set_title('angular velocity q1_dot')
    axs[3].set_ylabel("True/False")  # .set_ylabel("angular velocity, rpm")

    axs[4].plot(range(total), s_hist[:, 1], color='blue')
    axs[4].set_title('Expected Contact')  # .set_title('angular velocity q1_dot')
    axs[4].set_ylabel("True/False")  # .set_ylabel("angular velocity, rpm")

    plt.show()


def posplot(p_ref, p_hist):

    plt.plot(p_hist[:, 0], p_hist[:, 1], color='blue', label='body position')
    plt.title('Body XZ Position')
    plt.ylabel("z (m)")
    plt.xlabel("x (m)")
    plt.scatter(0, 0, color='green', marker="x", s=100, label='starting position')
    plt.scatter(p_ref[0], p_ref[1], color='orange', marker="x", s=100, label='position setpoint')
    plt.legend(loc="upper left")

    plt.show()


def posplot3d(p_ref, p_hist, total):
    ax = plt.axes(projection='3d')
    ax.plot(p_hist[:, 0], range(total), p_hist[:, 1], color='blue', label='body position')
    ax.set_title('Body XZ Position')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("time (s)")
    ax.set_zlabel("z (m)")
    ax.plot(np.zeros(total), range(total), np.zeros(total), color='green', label='starting position')
    pref0 = list(itertools.repeat(p_ref[0], total))
    pref1 = list(itertools.repeat(p_ref[1], total))
    ax.plot(pref0, range(total), pref1, color='orange', label='position setpoint')
    # ax.legend(loc="upper left")

    plt.show()