import numpy as np
import time

class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()

    def __exit__(self, type, value, traceback):
        print("Elapsed time: {:.3f} sec".format(time.time() - self._startTime))

# ###############################################################
# ##################MATRIX MULT#############################
# ###############################################################
# def transp(matr):
#     return [[matr[j][i] for j in range(len(matr))] for i in range(len(matr[0]))]
#
# def no_numpy_mult(first, second):
#     f1 = lambda a, b: [a[i]*b[i] for i in range(len(a))]
#     result = ([[sum(f1(i, j)) for j in transp(second)] for i in first])
#     return result
#
# def no_numpy_mult2(first, second):
#     result = []
#     for i in range(len(first)):
#         b_l = []
#         for j in range(len(second[0])):
#             s = 0
#             for k in range(len(second)):
#                 s += first[i][k]*second[k][j]
#             b_l.append(s)
#         result.append(b_l)
#     return result
#
#
# def numpy_mult(first, second):
#     result = np.dot(first, second)
#     return result
#
# x = np.random.sample((2, 2))
# y = np.random.sample((2, 2))
#
# with Profiler() as p1:
#     M1 = numpy_mult(np.array(x), np.array(y))
# with Profiler() as p2:
#     M2 = no_numpy_mult(x, y)
# with Profiler() as p3:
#     M3 = no_numpy_mult2(x, y)
#
# print(np.abs(np.array(M1) - M2).sum())
# print(np.abs(np.array(M1) - M3).sum())

# ###############################################################
# ##################SCALAR MULT#############################
# ###############################################################
# def no_numpy_scalar(v1, v2):
#     return sum([v1[i]*v2[i] for i in range(len(v1))])
#
# def numpy_scalar (v1, v2):
#     return np.matmul(v1, v2)
#
# x = [1, 2, 3]
# y = [4, 5, 6]
# print(numpy_scalar(x, y))
# print(no_numpy_scalar(x, y))
# ###############################################################
# #############ODD DIAG ELS SUM###########################
# ###############################################################
# def diag_2k(a):
#     a = a*np.eye(len(a))
#     return sum(a[a % 2 == 0])
#
# mat = np.random.randint(1, 10, size=(5, 5))
# print(mat)
# print(diag_2k(mat))
# ###############################################################
# #############PARTIAL SUMS###########################
# ###############################################################
# def cumsum(A):
#     return [list(np.cumsum(el)) for el in A]
#
# mat = np.random.randint(1, 10, size=(5, 5))
# print(mat)
# print(cumsum(mat))
# ###############################################################
# #############PARTIAL SUMS###########################
# ###############################################################
# def encode(a):
#     result = [[], []]
#     prev = a[0]
#     c = 0
#     for el in a:
#         if el == prev:
#             c += 1
#         else:
#             result[0].append(prev)
#             result[1].append(c)
#             prev = el
#             c = 1
#     result[0].append(prev)
#     result[1].append(c)
#     return result
#
# def encode_only_numpy(a):
#     starts = np.r_[0, np.where(~np.isclose(a[1:], a[:-1]))[0] + 1]
#     return np.take(a, starts), np.diff(np.r_[starts, len(a)])
#
# mat = np.array([1, 2, 2, 3, 3, 1, 1, 5, 5, 2, 3, 3])
# print(mat)
# print(encode(mat))
# ###############################################################
# #############SOME TRANSFORMATION###########################
# ###############################################################
def transform_Local(X, a=1):
    X1 = X.copy()
    X2 = X.copy()
    X1[[i % 2 == 0 for i in range(len(X))]] = 0
    X2[[not i % 2 == 0 for i in range(len(X))]] = 0
    return np.append(X, np.add([a*(not i == 0) for i in X1], [i**3 for i in X2])[::-1])

def transform(X, a=1):
    return [transform_Local(el, a) for el in X]

mat = np.array([100,200,300,400,500])
print(mat)
print(transform_Local(mat))
