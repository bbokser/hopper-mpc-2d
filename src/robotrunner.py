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
import itertools

np.set_printoptions(suppress=True, linewidth=np.nan)


class Runner:
    def __init__(self, dims='2d', ctrl='mpc', dt=1e-3):
        self.dims = dims
        self.ctrl = ctrl
        self.dt = dt
        self.total_run = 5000
        self.tol = 1e-3  # desired mpc tolerance
        self.hconst = 0.3
        self.n_x = 5  # number of states
        self.n_u = 2  # number of controls
        self.m = 6  # mass of the robot
        self.N = 10  # mpc horizon length
        mu = 0.3  # coeff of friction
        self.g = 9.81  # gravitational acceleration, m/s2
        self.t_p = 1  # gait period, seconds
        self.phi_switch = 0.5  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
        # for now, mpc sampling time is equal to gait period
        self.mpc_t = copy.copy(self.t_p*self.phi_switch)  # mpc sampling time
        self.mpc = mpc_cvx.Mpc(t=self.mpc_t, N=self.N, m=self.m, g=self.g, mu=mu)

        self.sh = 1  # estimated contact state

        self.pos_ref = np.array([1, 1])  # desired body position in world coords
        self.vel_ref = np.array([0, 0])  # desired body velocity in world coords
        self.X_ref = np.hstack([self.pos_ref, self.vel_ref, self.g]).T  # desired state

    def run(self):
        total = self.total_run + 1  # number of timesteps to plot
        t = 0  # time
        t0 = t  # starting time

        mpc_factor = self.mpc_t / self.dt  # repeat mpc every x seconds
        mpc_counter = mpc_factor
        force_f = None

        X_traj = np.zeros((total, self.n_x))
        f_hist = np.zeros((total, self.n_u))
        s_hist = np.zeros((total, 2))

        X_traj[0, :] = np.array([0, 0, 0, 0, self.g])  # initial conditions

        sh = 0

        for k in range(0, self.total_run):
            t = t + self.dt

            s = self.gait_scheduler(t, t0)

            if mpc_counter == mpc_factor:  # check if it's time to restart the mpc
                mpc_counter = 0  # restart the mpc counter
                # if np.linalg.norm(X_traj[k, :] - self.X_ref) > self.tol:  # then check if the error is high enough
                force_f, sh = self.mpc.mpcontrol(X_in=X_traj[k, :], X_ref=self.X_ref, s=s)  # take first timestep only

            mpc_counter += 1

            s_hist[k, :] = [s, sh]

            if self.ctrl == 'mpc':
                f_hist[k, :] = force_f[:, 0]

            else:  # Open loop traj opt
                if k == 0:
                    j = int(self.total_run/self.N)
                    print("j = ", j)
                    for i in range(0, self.N):
                        f_hist[int(i*j):int(i*j+j), :] = list(itertools.repeat(force_f[:, i], j))

            X_traj[k+1, :] = self.rk4(xk=X_traj[k, :], uk=f_hist[k, :])
            # X_traj[k + 1, :] = self.dynamics_dt(X=X_traj[k, :], U=f_hist[k, :])

        # print(X_traj[-1, :])
        # print(f_hist[4500, :])
        plots.fplot(total, p_hist=X_traj[:, 0:2], f_hist=f_hist, s_hist=s_hist)
        plots.posplot3d(p_ref=self.X_ref[0:2], p_hist=X_traj[:, 0:2], total=total)

        return None

    def dynamics_ct(self, X, U):
        # CT dynamics X -> dX
        m = self.m
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, -1],
                      [0, 0, 0, 0, 0]])
        B = np.array([[0, 0],
                      [0, 0],
                      [1 / m, 0],
                      [0, 1 / m],
                      [0, 0]])
        X_next = A @ X + B @ U
        return X_next

    def dynamics_dt(self, X, U):
        m = self.m
        t = self.dt
        n_x = self.n_x  # number of states
        n_u = self.n_u  # number of controls
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, -1],
                      [0, 0, 0, 0, 0]])
        B = np.array([[0, 0],
                      [0, 0],
                      [1 / m, 0],
                      [0, 1 / m],
                      [0, 0]])
        AB = np.vstack((np.hstack((A, B)), np.zeros((n_u, n_x + n_u))))
        M = expm(AB * t)
        Ad = M[0:n_x, 0:n_x]
        Bd = M[0:n_x, n_x:n_x + n_u]
        X_next = Ad @ X + Bd @ U
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

