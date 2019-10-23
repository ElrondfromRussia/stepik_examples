def numerical_derivative(func):
    h = 0.00001
    def deriv_func(x):
        return ((func(x+h) - func(x))/h)
    return deriv_func

def grad_descent0(funbc, deriv, good_point):
    start = good_point
    f_cur = funbc(start)
    eps = 0.000000001
    h = 0.05
    iters = 1000000
    for i in range(iters):
        start = start - h*deriv(start)
        if math.fabs(funbc(start) - f_cur) < eps:
            break
        f_cur = funbc(start)
    return start

import math
import numpy as np

def grad_descent(func, deriv):
    start = 1.0
    eps = 0.000000001
    h = 0.05
    iters = 100000
    starts = np.linspace(-3, 3, 10)
    min_x = start
    min_f = func(start)
    for st in starts:
        f_cur = func(st)
        for i in range(iters):
            st = st - h * deriv(st)
            if math.fabs(func(st) - f_cur) < eps:
                break
            f_cur = func(st)
        if func(st) < min_f:
            min_f = func(st)
            min_x = st
    return min_x


# ###################################

def mcos(x):
    return math.sin(x)

def x_qv(x):
    return x**2
def x_qv_dev(x):
    return 2*x
def x_f2(x):
    return x**2 + math.cos(x**3)
def x_f2_der(x):
    return 2*x - math.sin(x**3)*3*(x**2)

#mf = numerical_derivative(mcos)
#print(mf(0))
#print(grad_descent(x_qv, x_qv_dev, 1))
print(grad_descent(x_qv, x_qv_dev))
print(grad_descent(x_f2, x_f2_der))

#####################################################
#####################################################
#####################################################
#use numpy arrays as vectors. Do not use lists or tuples. It will be counted as invalid solution

def num_deriv(func):
    h = 0.00001
    def deriv_func(args):
        return (func(np.array([args[0] + h, args[1] + h])) - func(args)) / h
    return deriv_func

def grad_descent(func):
    func_deriv = num_deriv(func)
    h = 0.05
    iters = 100000
    eps = 0.000000001
    starts_x = np.array([np.linspace(-5, 5, 10)])
    starts_y = np.array([np.linspace(-5, 5, 10)])
    for ind, st_x in enumerate(starts_x):
        f_cur = func(np.array([st_x, starts_y[ind]]))
        for i in range(iters):
            st = st - h * func_deriv(st)
            if math.fabs(func(st) - f_cur) < eps:
                break
            f_cur = func(st)
    return 0
