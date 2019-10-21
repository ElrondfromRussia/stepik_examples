import numpy as np

# A = np.array([[1, 0], [0, 1]])
# B = np.array([[4, 1], [2, 2]])
# b = np.dot(A, B)
# print(b)
#
#
# y = np.array([1, 0])
# z = np.dot(A, y)
# x = np.linalg.solve(A, z)
# print(x)
#
# from scipy.optimize import minimize
#
# def f(x):
#     return x ** 2
#
# import matplotlib.pyplot as plt
#
# x = np.arange(-3, 3, .1)
# y = f(x)
#
# plt.plot(x,y)
# #plt.show()
#
# res = minimize(f, x0=100)
# print(res)
#
# from scipy.integrate import quad, odeint
# from scipy.special import erf
#
# def f(x):
#     return np.exp(-x ** 2)
#
#
# x = np.arange(-3, 3, .1)
# y = f(x)
#
# plt.plot(x,y);
# #plt.show()
# res, err = quad(f, 0, np.inf)
# print(np.sqrt(np.pi) / 2, res, err)
# res, err = quad(f, 0, 1)
# print(np.sqrt(np.pi) / 2 * erf(1), res, err)

# ############################################
from PIL import Image
import requests
from io import BytesIO
from matplotlib import pyplot as plt

url = 'https://s8.hostingkartinok.com/uploads/images/2018/08/308b49fcfbc619d629fe4604bceb67ac.jpg'

response = requests.get(url)
img = Image.open(BytesIO(response.content))
print(type(img))
img = np.array(img)
print(img.shape)
#plt.imshow(img)
img2 = img[:, :, ::-1]
print(img2.shape)
plt.imshow(img2)
img3 = img[:, ::-1]
plt.imshow(img3)
print(img3.shape)
batch = np.concatenate([img[None, :, :, :], img2[None, :, :, :], img3[None, :, :, :]])
print(batch.shape)
img4 = img.sum(axis=2)
plt.imshow(img4, cmap=plt.cm.gray)
print(img4.shape)
plt.show()
