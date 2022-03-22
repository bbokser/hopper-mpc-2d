"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import cvxpy as cp
from scipy.linalg import expm

class Mpc:

    def __init__(self, t, N, m, mu, g, **kwargs):
        self.t = t  # sampling time (s)
        self.N = N  # prediction horizon
        self.m = m  # kg
        self.mu = mu  # coefficient of friction
        self.g = g

    def mpcontrol(self, X_in, X_ref):
        N = self.N
        t = self.t
        m = self.m
        g = self.g
        mu = self.mu

        n_states = 4  # number of states
        n_controls = 2  # number of controls
        X = cp.Variable((n_states, N+1))
        U = cp.Variable((n_controls, N))

        A = np.vstack((np.hstack((np.zeros((2, 2)), np.eye(2))), np.zeros((2, 4))))
        B = np.vstack((np.zeros((2, 2)), np.eye(2) / m))
        G = np.array([0, 0, 0, -g]).T
        AB = np.vstack((np.hstack((A, B)), np.zeros((2, 6))))
        M = expm(AB*t)
        # M = AB @ AB * (1 + t ** 2) / 2 + AB * t + np.eye(np.shape(AB)[0])
        Ad = M[0:4, 0:4]
        Bd = M[0:4, 4:6]

        Q = np.eye(n_states)  # TODO: play around with this
        R = np.eye(n_controls)*0.25  # TODO: play around with this
        cost = 0
        constr = []

        # --- calculate cost & constraints --- #
        for k in range(0, N):
            cost += cp.quad_form(X[:, k+1] - X_ref, Q) + cp.quad_form(U[:, k], R)
            fx = U[0, k]
            fz = U[1, k]
            if (k % 2) == 0:  # even
                constr += [X[:, k + 1] == Ad @ X[:, k] + Bd @ U[:, k] + G]
            else:  # odd
                constr += [X[:, k + 1] == Ad @ X[:, k] + Bd @ U[:, k] + G]
                '''
                constr += [X[:, k + 1] == Ad @ X[:, k] + Bd @ U[:, k] + G,
                           0 >= fx - mu * fz,
                           0 >= -(fx + mu * fz),
                           0 <= fz]
                           '''

        constr += [X[:, 0] == X_in, X[:, N] == X_ref]  # initial and final condition
        # constr += [X[:, 0] == X_in]  # initial condition
        # --- set up solver --- #
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.OSQP)  # , verbose=True)
        u = np.zeros((2, N)) if U.value is None else U.value
        # print(X.value)
        # breakpoint()
        return u
