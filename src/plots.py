"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import matplotlib.pyplot as plt


def fplot(total, p_hist, f_hist):

    fig, axs = plt.subplots(3, sharex="all")
    plt.xlabel("Timesteps")

    axs[0].plot(range(total), p_hist[:, 1], color='blue')
    axs[0].set_title('Base z position w.r.t. time')
    axs[0].set_ylabel("z position (m)")

    axs[1].plot(range(total), f_hist[:, 0], color='blue')
    axs[1].set_title('Magnitude of X Output Force')
    axs[1].set_ylabel("Reaction Force Fx, N")

    axs[2].plot(range(total), f_hist[:, 1], color='blue')
    axs[2].set_title('Magnitude of Z Output Force')  # .set_title('angular velocity q1_dot')
    axs[2].set_ylabel("Reaction Force Fz, N")  # .set_ylabel("angular velocity, rpm")

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
