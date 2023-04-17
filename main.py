# system optimization based on genetic algorithm

# import libraries

import numpy as np

# define the coefficient matrix of two dimensions
c1 = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [
              0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], ])
c2 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], ])
c3 = np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], ])

# define F0 function


def F0(x, c):
    # x: input matrix of two dimensions
    # c: coefficient matrix of two dimensions
    # sum of the product of the input matrix and the coefficient matrix

    sum = np.sum(np.multiply(x, c))
    return sum

# define F01 function that is the same as F0 function with fixed coefficient matrix


def F01(x):
    return F0(x, c1)

# define F02 function that is the same as F0 function with fixed input matrix


def F02(x):
    return F0(x, c2)

# define F03 function that is the same as F0 function with fixed input matrix and coefficient matrix


def F03(x):
    return F0(x, c3)


# define some examples of input matrix
x1 = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], ])
x2 = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], ])
x3 = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], ])

# calculate the output of the function
print(F01(x1))
print(F02(x2))
print(F03(x3))
