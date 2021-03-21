from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import helper


def null_vector():
  # MISSING: Create a vector of size 10 and of type np.float32 full of zeros.
  return np.zeros(10, dtype=np.float32)


def chess():
  # MISSING: Create a 8x8 matrix and fill it with a checkerboard pattern
  # starting with 1 in the upper left and finishing with 0.
  # The matrix type should be np.int32.
  return np.array([[1 if (i+j)%2==0 else 0 for i in range(8)] for j in range(8)], dtype=np.int32)

def polar(a):
  # MISSING: "a" is a Nx2 matrix representing cartesian coordinates,
  # convert them to polar coordinates.
  # This function should return a Nx2 matrix where:
  # - the first column is the radius.
  #  -the second column is the angle.
  return np.transpose([(a[:,0]**2+a[:,1]**2)**(1/2), np.arctan2(a[:,1], a[:, 0])])


def cap(a, b):
  # MISSING: Cap all elements of "a" at "b" (i.e., max(a_{i,j}) <= b).
  # BONUS: Make sure to do so without modifying the input arguments.
  return np.minimum(a, b)


def moving_average(a):
  # MISSING: Compute the moving average of the elements of "a" over a
  # window of size 3.
  return [(a[i-1]+a[i]+a[i+1])/3 for i in range(1, len(a)-1)]


def main():
  helper.check_solution(null_vector, [0] * 10)
  helper.check_solution(chess, [[1, 0, 1, 0, 1, 0, 1, 0],
                                [0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1, 0],
                                [0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1, 0],
                                [0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1, 0],
                                [0, 1, 0, 1, 0, 1, 0, 1]])
  helper.check_solution(polar, [[0.70710678118, np.pi / 4.],
                                [0.70710678118 * 2, np.pi / 4.]],
                        np.array([[.5, .5], [1., 1.]], np.float32),
                        approx=True)
  helper.check_solution(cap, [1, 2, 2], np.array([1, 2, 3]), 2)
  helper.check_solution(cap, [2, 1, 2], np.array([3., 1., 3.]), 2)
  helper.check_solution(moving_average, [2, 3], np.array([1., 2., 3., 4.]))
  helper.check_solution(moving_average, [2], np.array([2., 2., 2.]))


if __name__ == '__main__':
  main()
