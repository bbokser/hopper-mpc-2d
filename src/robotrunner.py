"""
Copyright (C) 2020-2022 Benjamin Bokser
"""
import plots
import mpc_cvx

# import time
# import sys
import numpy as np
from scipy.linalg import expm
import itertools
np.set_printoptions(suppress=True, linewidth=np.nan)
from copy import copy
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
import sys

def projection(p0, v):
    # find point p projected onto ground plane from point p0 by vector v
    z = 0
    t = (z - p0[2]) / v[2]
    x = p0[0] + t * v[0]
    y = p0[1] + t * v[1]
    p = np.array([x, y, z])
    return p


class Runner:
    def __init__(self, ctrl='mpc', dt=1e-3):
        self.ctrl = ctrl
        self.dt = dt
        self.tol = 1e-3  # desired mpc tolerance
        self.m = 7.5  # mass of the robot, kg
        self.N = 10  # mpc horizon length
        self.g = 9.81  # gravitational acceleration, m/s2
        self.ref_curve = False
        self.t_p = 0.8  # gait period, seconds
        self.phi_switch = 0.5  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
        # for now, mpc sampling time is equal to gait period
        self.mpc_dt = self.t_p * self.phi_switch  # mpc sampling time
        self.N_time = self.N * self.mpc_dt  # mpc horizon time

        self.n_x = 6  # number of states
        self.n_u = 3  # number of controls
        self.A = np.array([[0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])
        self.B = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [1 / self.m, 0, 0],
                           [0, 1 / self.m, 0],
                           [0, 0, 1 / self.m]])
        self.G = np.array([0, 0, 0, 0, 0, -self.g])
        self.X_0 = np.array([0, 0, 0.35, 0, 0, 0])
        self.X_f = np.array([2, 2, 0.35, 0, 0, 0])  # desired final state

        mu = 1  # coeff of friction
        self.mpc = mpc_cvx.Mpc(t=self.mpc_dt, A=self.A, B=self.B, G=self.G.reshape((-1, 1)), N=self.N, m=self.m, g=self.g, mu=mu)

        if self.ctrl == 'openloop':
            self.mpc_factor = int(self.mpc_dt / self.dt)  # repeat mpc every x low level timesteps
            self.t_run = int(self.N * self.mpc_factor)  # override runtime for open loop traj opt
        else:
            self.mpc_factor = int(2 * self.mpc_dt / self.dt)  # repeat mpc every x low level timesteps
            self.t_run = 5000

        self.N_time = self.N * self.mpc_dt  # mpc horizon time
        self.N_k = int(self.N * self.mpc_factor)  # total mpc prediction horizon length (low-level timesteps)
        # self.t_start = 0.5 * self.t_p * self.phi_switch  # start halfway through stance phase
        self.t_start = 0  # can't start halfway throughs stance due to mpc timestep size = phase
        self.t_st = self.t_p * self.phi_switch  # time spent in stance
        self.Nc = self.t_st / self.dt  # number of timesteps spent in contact
        self.ref_spline = None
        # self.N_mpc = self.mpc_factor / self.N  # low level timesteps per mpc timestep
        print('total_run = ', self.t_run)

    def run(self):
        total = self.t_run + 1  # number of timesteps to plot
        t = copy(self.t_start)  # time
        # t0 = t  # starting time
        mpc_factor = self.mpc_factor  # repeat mpc every x seconds
        mpc_counter = copy(mpc_factor)
        X_traj = np.zeros((total, self.n_x))
        X_traj[0, :] = self.X_0  # initial conditions
        f_hist = np.zeros((total, self.n_u))
        s_hist = np.zeros(total)
        U_pred = np.zeros((self.N, self.n_u))
        if self.ctrl == 'openloop':
            pf_ref = np.zeros((self.N+1, self.n_u))
        else:
            pf_ref = np.zeros(self.n_u)

        j = int(self.mpc_factor)
        X_pred_hist = np.zeros((self.N+1, self.n_u))
        f_pred_hist = np.zeros((self.N+1, self.n_u))
        p_pred_hist = np.zeros((self.N+1, self.n_u))

        X_ref, C = self.ref_traj_init(X_in=X_traj[0, :], X_f=self.X_f)

        for k in range(0, self.t_run):
            # s = self.gait_scheduler(t, t0)  # not really necessary now that we have C[k]...
            if self.ctrl == 'mpc':
                if mpc_counter == mpc_factor:  # check if it's time to restart the mpc
                    mpc_counter = 0  # restart the mpc counter
                    X_refN = self.ref_traj_grab(X_ref, k, factor=int(self.mpc_dt / self.dt))
                    Ck = self.Ck_grab(C, k, factor=self.mpc_factor)  # TODO: Check
                    U_pred, X_pred = self.mpc.mpcontrol(X_in=X_traj[k, :], X_ref=X_refN, Ck=Ck)
                    # p_pred = (X_pred[2, 0:3]+(X_pred[2, 0:3]+X_pred[3, 0:3])/2)/2  # next pred body pos over next step
                    # p_pred = (X_pred[2, 0:3] + X_pred[3, 0:3]) / 2  # next pred body pos over next ftstep
                    p_pred = X_pred[3, 0:3]
                    f_pred = U_pred[2, :]  # next predicted foot force vector
                    p_pred_hist = np.vstack((p_pred_hist, p_pred))
                    f_pred_hist = np.vstack((f_pred_hist, 0.5*f_pred/np.sqrt(np.sum(f_pred**2))))
                    pf_ref = np.vstack((pf_ref, projection(p_pred, f_pred)))
                    X_pred_hist = np.dstack((X_pred_hist, X_pred[:, 0:self.n_u]))
                mpc_counter += 1
                f_hist[k, :] = U_pred[0, :]  # take first timestep

            else:  # Open loop traj opt, this will fail if total != mpc_factor
                if k == 0:
                    X_refN = self.ref_traj_grab(X_ref, k, factor=self.mpc_factor)
                    Ck = self.Ck_grab(C, k, factor=self.mpc_factor)
                    U_pred, X_pred = self.mpc.mpcontrol(X_in=X_traj[k, :], X_ref=X_refN, Ck=Ck)
                    for i in range(0, self.N):
                        f_hist[int(i*j):int(i*j+j), :] = list(itertools.repeat(U_pred[i, :], j))
                    p_pred_hist = X_pred[:-1, 0:3]
                    f_pred_hist = np.array([0.5 * U_pred[i, :] / np.sqrt(np.sum(U_pred[i, :] ** 2)) * Ck[i]
                                           for i in range(self.N)])
            s_hist[k] = C[k]
            X_traj[k+1, :] = self.rk4(xk=X_traj[k, :], uk=f_hist[k, :])
            t = t + self.dt
            # X_traj[k + 1, :] = self.dynamics_dt(X=X_traj[k, :], U=f_hist[k, :], t=self.dt)

        # print(X_traj[-1, :])
        # print(f_hist[4500, :])
        plots.posfplot(p_ref=X_ref, p_hist=X_traj[:, 0:self.n_u], p_pred_hist=p_pred_hist, f_pred_hist=f_pred_hist,
                       pf_hist=pf_ref)
        plots.posplot_animate(p_f=self.X_f[0:3], p_hist=X_traj[::50, 0:self.n_u], ref_traj=X_ref[::50, :],
                              p_pred_hist=p_pred_hist, f_pred_hist=f_pred_hist)  # pf_ref=pf_ref)
        plots.fplot(total, p_hist=X_traj[:, 0:self.n_u], f_hist=f_hist, s_hist=s_hist)
        return None

    def dynamics_ct(self, X, U):
        # CT dynamics X -> dX
        A = self.A
        B = self.B
        G = self.G
        X_next = A @ X + B @ U + G
        return X_next

    def dynamics_dt(self, X, U, t):
        n_x = self.n_x  # number of states
        n_u = self.n_u  # number of controls
        A = self.A
        B = self.B
        G = self.G
        ABG = np.hstack((A, B, G))
        ABG.resize((n_x + n_u + 1, n_x + n_u + 1))
        M = expm(ABG * t)
        Ad = M[0:n_x, 0:n_x]
        Bd = M[0:n_x, n_x:n_x + n_u]
        Gd = M[0:n_x, -1]
        X_next = Ad @ X + Bd @ U + Gd
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

    def phase_remainder(self, t, t0):
        phi = np.mod((t - t0) / self.t_p, 1)  # percentage of phase
        return phi * self.t_p

    def contact_map(self, N, dt, ts, t0):
        # generate vector of scheduled contact states over the mpc's prediction horizon
        C = np.zeros(N)
        for k in range(0, N):
            C[k] = self.gait_scheduler(t=ts, t0=t0)
            ts += dt
        return C

    def ref_traj_init(self, X_in, X_f):
        # Path planner--generate low-level reference trajectory for the entire run
        N_k = self.N_k  # total MPC horizon in low-level timesteps
        t_run = self.t_run
        dt = self.dt
        t_sit = 0  # timesteps spent "sitting" at goal
        t_traj = int(t_run - t_sit)  # timesteps for trajectory not including sit time
        t_ref = t_run + N_k  # timesteps for reference (extra for MPC)
        x_ref = np.linspace(start=X_in, stop=X_f, num=t_traj)  # interpolate positions
        C = self.contact_map(t_ref, dt, self.t_start, 0)  # low-level contact map for the entire run
        if self.ref_curve is True:
            spline_t = np.array([0, t_traj * 0.3, t_traj])
            spline_y = np.array([X_in[1], X_f[1] * 0.7, X_f[1]])
            csy = CubicSpline(spline_t, spline_y)
            for k in range(t_traj):
                x_ref[k, 1] = csy(k)  # create evenly spaced sample points of desired trajectory

        x_ref = np.vstack((x_ref, np.tile(X_f, (N_k + t_sit, 1))))  # sit at the goal
        period = self.t_p  # *1.2  # * self.mpc_dt / 2
        amp = self.t_p / 4  # amplitude
        phi = np.pi*3/2  # phase offset
        # make height sine wave
        sine_wave = np.array([X_in[2] + amp + amp * np.sin(2 * np.pi / period * (i * dt) + phi) for i in range(t_ref)])
        peaks = find_peaks(sine_wave)[0]
        troughs = find_peaks(-sine_wave)[0]
        spline_k = np.sort(np.hstack((peaks, troughs)))  # independent variable
        spline_k = np.hstack((0, spline_k))  # add initial footstep idx based on first timestep
        spline_k = np.hstack((spline_k, t_ref - 1))  # add final footstep idx based on last timestep
        n_k = np.shape(spline_k)[0]
        spline_i = np.zeros((n_k, 3))
        spline_i[:, 0:2] = x_ref[spline_k, 0:2]
        spline_i[:, 2] = sine_wave[spline_k]  # dependent variable
        ref_spline = CubicSpline(spline_k, spline_i, bc_type='clamped')  # generate cubic spline
        x_ref[:, 0:3] = [ref_spline(k) for k in range(t_ref)]  # create z-spline

        x_ref[:-1, 3:6] = [(x_ref[i + 1, 0:3] - x_ref[i, 0:3]) / dt for i in range(t_ref - 1)]  # interpolate linear vel
        # np.set_printoptions(threshold=sys.maxsize)
        return x_ref, C

    def ref_traj_grab(self, ref, k, factor):  # Grab appropriate timesteps of pre-planned trajectory for mpc
        return ref[k:(k + self.N_k):factor, :]  # change to mpc-level timesteps

    def Ck_grab(self, C, k, factor):  # Grab appropriate timesteps of pre-planned trajectory for mpc
        C_ref = C[k:(k + self.N_k)]
        # Ck = np.zeros(self.N)
        Ck = np.array([np.median(C_ref[(i * factor):(i * factor + factor)]) for i in range(self.N)])
        return Ck  # change to mpc-level timesteps


