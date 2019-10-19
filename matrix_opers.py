import numpy as np

import time

class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()

    def __exit__(self, type, value, traceback):
        print("Elapsed time: {:.3f} sec".format(time.time() - self._startTime))

def transp(matr):
    return [[matr[j][i] for j in range(len(matr))] for i in range(len(matr[0]))]

def no_numpy_mult(first, second):
    f1 = lambda a, b: [a[i]*b[i] for i in range(len(a))]
    result = ([[sum(f1(i, j)) for j in transp(second)] for i in first])
    return result

def no_numpy_mult2(first, second):
    result = []
    for i in range(len(first)):
        b_l = []
        for j in range(len(second[0])):
            s = 0
            for k in range(len(second)):
                s += first[i][k]*second[k][j]
            b_l.append(s)
        result.append(b_l)
    return result


def numpy_mult(first, second):
    result = np.dot(first, second)
    return result


# x = [
#         [1.0, 2], [4.0, 5]
#     ]
# y = [
#         [1, 2, 3, 6], [4, 5.0, 3, 6]
#     ]

x = np.random.sample((2, 2))
y = np.random.sample((2, 2))

with Profiler() as p1:
    M1 = numpy_mult(np.array(x), np.array(y))
with Profiler() as p2:
    M2 = no_numpy_mult(x, y)
with Profiler() as p3:
    M3 = no_numpy_mult2(x, y)

print(np.abs(np.array(M1) - M2).sum())
print(np.abs(np.array(M1) - M3).sum())

# ###############################################################
# ###############################################################
# ###############################################################
