"""
Copyright (C) 2020-2022 Benjamin Bokser

Reference Material:
https://www.youtube.com/watch?v=RrnkPrcpyEA
"""

import numpy as np
import itertools
import casadi as cs

class Mpc:

    def __init__(self, t, N, m, mu, g, **kwargs):

        self.t = t  # sampling time (s)
        self.N = N  # prediction horizon
        # horizon length = self.dt*self.N = .25 seconds
        self.m = m  # kg
        self.mu = mu  # coefficient of friction
        self.g = g

    def mpcontrol(self, X_in, X_ref):
        N = self.N
        t = self.t
        m = self.m
        g = self.g

        x = cs.SX.sym('x')
        z = cs.SX.sym('z')
        dx = cs.SX.sym('dx')
        dz = cs.SX.sym('dz')
        states = [x, z, dx, dz]  # state vector x
        n_states = len(states)  # number of states

        fx = cs.SX.sym('fx')  # controls
        fz = cs.SX.sym('fz')  # controls
        controls = [fx, fz]
        n_controls = len(controls)  # number of controls

        # x_next = np.dot(A, states) + np.dot(B, controls) + g  # the discrete dynamics of the system
        X_next = [1.0*dx*t + 0.5*fx*(t**2 + 1)/m + 1.0*x,
                  1.0*dz*t + 0.5*fz*(t**2 + 1)/m + 1.0*z, 1.0*dx + 1.0*fx*t/m,
                  1.0*dz + 1.0*fz*t/m - g]

        dynamics = cs.Function('fn', [x, z, dx, dz, fx, fz], X_next)  # nonlinear mapping of function f(x,u)

        U = cs.SX.sym('U', n_controls, N)  # decision variables, control action matrix
        st_ref = cs.SX.sym('st_ref', n_states)  # initial and reference states
        X = cs.SX.sym('X', n_states, (N + 1))  # represents the states over the opt problem.

        obj = 0  # objective function
        constr = []  # constraints vector  # TODO: Preallocate

        k = 10
        Q = np.zeros((n_states, n_states))  # state weighing matrix
        np.fill_diagonal(Q, [k, k, k, k])

        R = np.zeros((n_controls, n_controls))  # control weighing matrix
        np.fill_diagonal(R, [k/2, k/2])

        # compute objective and constraints
        for k in range(0, N):
            st = X[:, k]  # state
            con = U[:, k]  # control action
            # calculate objective
            obj = obj + cs.mtimes(cs.mtimes((st - st_ref).T, Q), st - st_ref) + cs.mtimes(cs.mtimes(con.T, R), con)
            # calculate dynamics constraint
            st_n_e = dynamics(st[0], st[1], st[2], st[3], con[0], con[1])
            constr = cs.vertcat(constr, X[:, k + 1] - np.array(st_n_e))

        # add additional constraints
        for k in range(0, N):
            constr = cs.vertcat(constr, U[0, k] - self.mu * U[1, k])  # fx - mu*fz
            constr = cs.vertcat(constr, -U[0, k] - self.mu * U[1, k])  # -fx - mu*fz

        opt_variables = cs.vertcat(cs.reshape(X, n_states * (N + 1), 1),
                                   cs.reshape(U, n_controls * N, 1))
        qp = {'x': opt_variables, 'f': obj, 'g': constr, 'p': st_ref}
        opts = {'print_time': 0, 'error_on_fail': 0, 'printLevel': "none", 'boundTolerance': 1e-6,
                'terminationTolerance': 1e-6}
        solver = cs.qpsol('S', 'qpoases', qp, opts)

        c_length = np.shape(constr)[0]
        o_length = np.shape(opt_variables)[0]

        lbg = list(itertools.repeat(-1e10, c_length))  # inequality constraints: big enough to act like infinity
        lbg[0:N] = itertools.repeat(0, N)  # dynamics equality constraint
        ubg = list(itertools.repeat(0, c_length))  # inequality constraints

        # constraints for optimization variables
        lbx = list(itertools.repeat(-1e10, o_length))  # input inequality constraints
        ubx = list(itertools.repeat(1e10, o_length))  # input inequality constraints
        st_len = n_states * (N + 1)
        lbx[(st_len + 1)::2] = [0 for i in range(N)]  # lower bound on all fz
        lbx[1:st_len:2] = [0.1 for i in range(N)]  # lower bound on all z
        ubx[1:st_len:2] = [1.5 for i in range(N)]  # upper bound on all z

        # setup is finished, now solve-------------------------------------------------------------------------------- #

        u0 = np.zeros((N, n_controls))  # 3 control inputs
        X0 = np.matlib.repmat(X_in, 1, N + 1).T  # initialization of the state's decision variables

        # parameters and xin must be changed every timestep
        parameters = cs.vertcat(X_in, X_ref)  # set values of parameters vector
        # init value of optimization variables
        x0 = cs.vertcat(np.reshape(X0.T, (n_states * (N + 1), 1)), np.reshape(u0.T, (n_controls * N, 1)))

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=parameters)

        solu = np.array(sol['X'][n_states * (N + 1):])
        # u = np.reshape(solu.T, (n_controls, N)).T  # get controls from the solution
        u = np.reshape(solu.T, (N, n_controls)).T  # get controls from the solution

        u_cl = u[:, 0]  # ignore rows other than new first row
        # ss_error = np.linalg.norm(x0 - x_ref)  # defaults to Euclidean norm
        # print("ss_error = ", ss_error)
        # print("Time elapsed for MPC: ", t1 - t0)

        return u_cl
