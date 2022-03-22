"""
Copyright (C) 2020-2022 Benjamin Bokser
"""
import plots
import mpc_cvx

# import time
# import sys
import numpy as np
import copy
from scipy.linalg import expm

np.set_printoptions(suppress=True, linewidth=np.nan)


class Runner:
    def __init__(self, dt=1e-3):
        self.dt = dt
        self.total_run = 20000
        self.tol = 1e-3  # desired mpc tolerance
        self.hconst = 0.3

        self.m = 6  # mass of the robot
        N = 10  # mpc horizon length
        mu = 0.3  # coeff of friction
        self.g = 9.81  # gravitational acceleration, m/s2
        self.t_p = 1  # gait period, seconds
        self.phi_switch = 0.5  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
        # for now, mpc sampling time is equal to gait period
        self.mpc_t = copy.copy(self.t_p*self.phi_switch)  # mpc sampling time
        self.mpc = mpc_cvx.Mpc(t=self.mpc_t, N=N, m=self.m, mu=mu, g=self.g)

        self.sh = 1  # estimated contact state

        self.pos_ref = np.array([1, 1])  # desired body position in world coords
        self.vel_ref = np.array([0, 0])  # desired body velocity in world coords
        self.X_ref = np.hstack([self.pos_ref, self.vel_ref]).T  # desired state

    def run(self):
        total = self.total_run + 1  # number of timesteps to plot
        t = 0  # time
        t0 = t  # starting time

        mpc_factor = self.mpc_t/ self.dt  # repeat mpc every x seconds
        mpc_counter = mpc_factor
        force_f = None

        X_traj = np.zeros((total, 4))
        f_hist = np.zeros((total, 2))

        X_traj[0, :] = np.array([0, 1, 0, 0])  # initial conditions

        for k in range(0, self.total_run):
            t = t + self.dt

            s = self.gait_scheduler(t, t0)

            if mpc_counter == mpc_factor:  # check if it's time to restart the mpc
                mpc_counter = 0  # restart the mpc counter
                # if np.linalg.norm(X_traj[k, :] - self.X_ref) > self.tol:  # then check if the error is high enough
                force_f = self.mpc.mpcontrol(X_in=X_traj[k, :], X_ref=self.X_ref)[:, 0]  # take first timestep only

            mpc_counter += 1

            f_hist[k, :] = force_f # *s
            # X_traj[k+1, :] = self.rk4(xk=X_traj[k, :], uk=f_hist[k, :])
            X_traj[k + 1, :] = self.dynamics_dt(X=X_traj[k, :], U=f_hist[k, :])

        print(X_traj[-1, :])
        print(f_hist[-1, :])
        plots.fplot(total, p_hist=X_traj[:, 0:2], f_hist=f_hist)
        plots.posplot(p_ref=self.X_ref[0:2], p_hist=X_traj[:, 0:2])

        return None

    def dynamics_ct(self, X, U):
        # CT dynamics X -> dX
        A = np.vstack((np.hstack((np.zeros((2, 2)), np.eye(2))), np.zeros((2, 4))))
        B = np.vstack((np.zeros((2, 2)), np.eye(2) / self.m))
        G = np.array([0, 0, 0, -self.g]).T
        X_next = A @ X + B @ U + G
        return X_next

    def dynamics_dt(self, X, U):
        # DT dynamics X -> dX
        t = self.dt
        A = np.vstack((np.hstack((np.zeros((2, 2)), np.eye(2))), np.zeros((2, 4))))
        B = np.vstack((np.zeros((2, 2)), np.eye(2) / self.m))
        G = np.array([0, 0, 0, -self.g]).T
        AB = np.vstack((np.hstack((A, B)), np.zeros((2, 6))))
        M = expm(AB * t)
        Ad = M[0:4, 0:4]
        Bd = M[0:4, 4:6]
        X_next = Ad @ X + Bd @ U + G
        return X_next

    def rk4(self, xk, uk):
        # RK4 integrator solves for new X
        dynamics = self.dynamics_ct
        h = self.dt
        f1 = dynamics(xk, uk)
        f2 = dynamics(xk + 0.5 * h * f1, uk)
        f3 = dynamics(xk + 0.5 * h * f2, uk)
        f4 = dynamics(xk + h * f3, uk)
        return xk + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

    def gait_scheduler(self, t, t0):
        phi = np.mod((t - t0) / self.t_p, 1)
        if phi > self.phi_switch:
            s = 0  # scheduled swing
        else:
            s = 1  # scheduled stance
        return s

