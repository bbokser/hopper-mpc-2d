"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import cvxpy as cp
from scipy.linalg import expm


class Mpc:

    def __init__(self, t, A, B, N, m, g, mu, **kwargs):
        self.t = t  # sampling time (s)
        self.A = A
        self.B = B
        self.N = N  # prediction horizon
        self.m = m  # kg
        self.mu = mu  # coefficient of friction
        self.g = g

    def mpcontrol(self, X_in, X_ref, s):
        N = self.N
        t = self.t
        m = self.m
        mu = self.mu
        A = self.A
        B = self.B
        n_x = np.shape(self.A)[1]
        n_u = np.shape(self.B)[1]
        X = cp.Variable((n_x, N+1))
        U = cp.Variable((n_u, N))

        AB = np.vstack((np.hstack((A, B)), np.zeros((n_u, n_x+n_u))))
        M = expm(AB*t)
        Ad = M[0:n_x, 0:n_x]
        Bd = M[0:n_x, n_x:n_x+n_u]

        Q = np.eye(n_x)
        Q[n_u, n_u] *= 0.01
        Q[n_u + 1, n_u + 1] *= 0.01
        Q[n_u + 2, n_u + 2] *= 0.01
        R = np.eye(n_u)*0
        cost = 0
        constr = []
        U_ref = np.zeros(n_u)
        U_ref[-1] = m * self.g
        # --- calculate cost & constraints --- #
        if n_x == 5:
            for k in range(0, N):
                kf = 3 if k == N - 1 else 1  # terminal cost
                kuf = 0 if k == N - 1 else 1  # terminal cost
                cost += cp.quad_form(X[:, k+1] - X_ref[k, :], Q * kf) + cp.quad_form(U[:, k] - U_ref, R * kuf)
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
        elif n_x == 7:
            for k in range(0, N):
                kf = 10 if k == N - 1 else 1  # terminal cost
                kuf = 0 if k == N - 1 else 1  # terminal cost
                cost += cp.quad_form(X[:, k+1] - X_ref[k, :], Q * kf) + cp.quad_form(U[:, k] - U_ref, R * kuf)
                z = X[2, k]
                fx = U[0, k]
                fy = U[1, k]
                fz = U[2, k]
                if ((k + s) % 2) == 0:  # even
                    constr += [X[:, k + 1] == Ad @ X[:, k] + Bd @ U[:, k],
                               0 == fx,  # fx
                               0 == fy,  # fy
                               0 == fz,
                               z >= 0]  # fz
                else:  # odd
                    constr += [X[:, k + 1] == Ad @ X[:, k] + Bd @ U[:, k],
                               0 >= fx - mu * fz,
                               0 >= -fx - mu * fz,
                               0 >= fy - mu * fz,
                               0 >= -fy - mu * fz,
                               z >= 0]
        constr += [X[:, 0] == X_in, X[:, N] == X_ref[-1, :]]  # initial and final condition
        # constr += [X[:, 0] == X_in]  # initial condition
        # --- set up solver --- #
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.OSQP)  # , verbose=True)
        u = np.zeros((n_u, N)) if U.value is None else U.value
        # print(X.value)
        # breakpoint()
        return u, (s % 2)
