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
    def deriv_func(argus):
        dx = (func(np.array([argus[0] + h, argus[1]])) - func(argus)) / h
        dy = (func(np.array([argus[0], argus[1] + h])) - func(argus)) / h
        return np.array([dx, dy])
    return deriv_func

def grad_descent(func):
    func_deriv = num_deriv(func)
    h = 0.5
    iters = 1000
    eps = 0.000000001
    starts_x = np.array(np.linspace(-5, 5, 10))
    starts_y = np.array(np.linspace(-5, 5, 10))
    min_xy = np.array([starts_x[0], starts_y[0]])
    min_f = func(min_xy)
    for ind in range(len(starts_x)):
        margs = np.array([starts_x[ind], starts_y[ind]])
        f_cur = func(margs)
        for i in range(iters):
            margs = margs - h * func_deriv(margs)
            if math.fabs((func(margs) - f_cur).sum()) < eps:
                break
            f_cur = func(margs)
        if func(margs) < min_f:
            min_xy = margs
            min_f = func(margs)
    return min_xy

def xy_qv(npr):
    return npr[0]**2 + npr[1]**2
print(grad_descent(xy_qv))
