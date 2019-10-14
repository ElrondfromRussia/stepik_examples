import itertools


def is_simple(i):
    k = 0
    for j in range(2, i):
        if i % j == 0:
            k = k + 1
    if k == 0:
        return True
    else:
        return False


def primes():
    a = 1
    while True:
        a += 1
        if is_simple(a):
            yield a


print(list(itertools.takewhile(lambda x : x <= 31, primes())))
# [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
