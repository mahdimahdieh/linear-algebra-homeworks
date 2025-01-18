import numpy as np
import matplotlib.pyplot as plt
# null space
from scipy.linalg import null_space
import sympy as sym

numcourses = [13, 4, 12, 3, 14, 13, 12, 9, 11, 7, 13, 11, 9, 2, 5, 7, 10, 0, 9, 7]
happiness = [70, 25, 54, 21, 80, 68, 84, 62, 57, 40, 60, 64, 45, 38, 51, 52, 58, 21, 75, 70]

# Design matrix
X = np.hstack((np.ones((20, 1)), np.array(numcourses, ndmin=2).T))
y = np.array(happiness, ndmin=2).T

# Left-inverse method
XT_X = X.T @ X
XT_y = X.T @ y
beta1 = np.linalg.inv(XT_X) @ XT_y

# QR decomposition
Q, R = np.linalg.qr(X)
QT_y = Q.T @ y
beta2 = np.linalg.solve(R, QT_y)

print('Betas from left-inverse: ')
print(np.round(beta1, 3))
print(' ')
print('Betas from QR with inv(R): ')
print(np.round(beta2, 3))
print(' ')