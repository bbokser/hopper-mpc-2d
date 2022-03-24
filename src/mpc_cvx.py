"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import cvxpy as cp
from scipy.linalg import expm


class Mpc:

    def __init__(self, t, N, m, g, mu, **kwargs):
        self.t = t  # sampling time (s)
        self.N = N  # prediction horizon
        self.m = m  # kg
        self.mu = mu  # coefficient of friction
        self.g = g

    def mpcontrol(self, X_in, X_ref, s):
        N = self.N
        t = self.t
        m = self.m
        mu = self.mu

        n_x = 5  # number of states
        n_u = 2  # number of controls
        X = cp.Variable((n_x, N+1))
        U = cp.Variable((n_u, N))

        A = np.array([[0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, -1],
                     [0, 0, 0, 0, 0]])
        B = np.array([[0, 0],
                      [0, 0],
                      [1/m, 0],
                      [0, 1/m],
                      [0, 0]])
        AB = np.vstack((np.hstack((A, B)), np.zeros((n_u, n_x+n_u))))
        M = expm(AB*t)
        Ad = M[0:n_x, 0:n_x]
        Bd = M[0:n_x, n_x:n_x+n_u]

        Q = np.eye(n_x)  # TODO: play around with this
        Q[2, 2] *= 0.1
        Q[3, 3] *= 0.1
        Q[4, 4] *= 0
        R = np.eye(n_u)*0.0  # TODO: play around with this
        cost = 0
        constr = []
        U_ref = np.array([0, m * self.g])
        # --- calculate cost & constraints --- #
        for k in range(0, N):
            kf = 3 if k == N - 1 else 1  # terminal cost
            # kuf = 0 if k == N - 1 else 1  # terminal cost
            cost += cp.quad_form(X[:, k+1] - X_ref, Q * kf) + cp.quad_form(U[:, k] - U_ref , R)
            fx = U[0, k]
            fz = U[1, k]
            if ((k + s) % 2) == 0:  # even
                constr += [X[:, k + 1] == Ad @ X[:, k] + Bd @ U[:, k],
                           0 == fx,  # fx
                           0 == fz]  # fz
            else:  # odd
                constr += [X[:, k + 1] == Ad @ X[:, k] + Bd @ U[:, k],
                           0 >= fx - mu * fz,
                           0 >= -(fx + mu * fz),
                           0 <= fz]

        constr += [X[:, 0] == X_in, X[:, N] == X_ref]  # initial and final condition
        # constr += [X[:, 0] == X_in]  # initial condition
        # --- set up solver --- #
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.OSQP)  # , verbose=True)
        u = np.zeros((n_u, N)) if U.value is None else U.value
        # print(X.value)
        # breakpoint()
        return u, (s % 2)
