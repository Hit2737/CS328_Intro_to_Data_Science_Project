import sys
from numpy.linalg import norm
from numpy import dot
import numpy as np

from syntheticDataMaker import SyntheticDataMaker
from frequentDirections import FrequentDirections

import matplotlib.pyplot as plt

n = 500
d = 50
ell = 20
k = 5

# Shape of sketch matrix is ell X d
# Shape of A is 1000 X 50 (if their code is unchanged)

# this is only needed for generating input vectors
dataMaker = SyntheticDataMaker()
dataMaker.initBeforeMake(d, k, signal_to_noise_ratio=10.0)                                                                                                                                                                                                                                                                                                                         

# This is where the sketching actually happens

sketcher = FrequentDirections(d,ell)

A = []
for i in range(n):
    row = dataMaker.makeRow()
    A.append(row)

for i in range(n):
    sketcher.append(A[i])
sketch = sketcher.get()

A = np.array(A)
print(A.shape)

# Here is where you do something with the sketch.
# The sketch is an ell by d matrix 
# For example, you can compute an approximate covariance of the input 
# matrix like this:

# print(sketch.shape)
approxCovarianceMatrix = dot(sketch.transpose(),sketch)
print(approxCovarianceMatrix.shape)
# print(approxCovarianceMatrix)
# print(approxCovarianceMatrix.shape)

import numpy as np

# Load the matrix A from the CSV file
# A = np.loadtxt('matrix.csv', delimiter=',')
# print("Shape of matrix A:", A.shape)
# Compute A^T . A
result = np.dot(A.T, A)

# print("Result of A^T . A:")
# print(result)
# print(result.shape)

# UNCOMMMENT FROM HERE

# frobenius_norm = np.linalg.norm(result - approxCovarianceMatrix, ord='fro')
# print(frobenius_norm)


# arr = []
# print("Frobenius norm between matrices AT.A and BT.B on performing simple sketching:", frobenius_norm)

# arr.append(frobenius_norm)

# for j in [2,3,4,5,6,7,8,9,10]:

#     sketcher = FrequentDirections(d,ell)
#     for i in range(0,n,j):
#         sketcher.append(A[i])
#     sketch = sketcher.get()

#     approxCovarianceMatrix = dot(sketch.transpose(),sketch)

#     frobenius_norm = np.linalg.norm(result - approxCovarianceMatrix, ord='fro')
#     arr.append(frobenius_norm)

# plt.plot(np.arange(1,11),arr)
# plt.show()

# print(sorted(np.random.choice(np.arange(1, 501), size=400, replace=False)))

# temp=0
# tarr = []
# for j in range(100):
#     sketcher = FrequentDirections(d,ell)
#     for i in sorted(np.random.choice(np.arange(0,500), size=400, replace=False)):
#         sketcher.append(A[i])
#     sketch = sketcher.get()
#     approxCovarianceMatrix = dot(sketch.transpose(),sketch)
#     frobenius_norm = np.linalg.norm(result - approxCovarianceMatrix, ord='fro')
#     temp+=frobenius_norm
#     tarr.append(frobenius_norm)

# temp/=100
# print("Average F norm on samling 400 rows",temp)
# plt.plot(np.arange(1,101),tarr)
# plt.show()

# temp=0
# tarr = []
# for j in range(100):
#     sketcher = FrequentDirections(d,ell)
#     for i in sorted(np.random.choice(np.arange(0,500), size=450, replace=False)):
#         sketcher.append(A[i])
#     sketch = sketcher.get()
#     approxCovarianceMatrix = dot(sketch.transpose(),sketch)
#     frobenius_norm = np.linalg.norm(result - approxCovarianceMatrix, ord='fro')
#     temp+=frobenius_norm
#     tarr.append(frobenius_norm)

# temp/=100
# print("Average F norm on sampling 450 rows:",temp)
# plt.plot(np.arange(1,101),tarr)
# plt.show()

# temp=0
# tarr = []
# for j in range(100):
#     sketcher = FrequentDirections(d,ell)
#     for i in np.random.permutation(500):
#         sketcher.append(A[i])
#     sketch = sketcher.get()
#     approxCovarianceMatrix = dot(sketch.transpose(),sketch)
#     frobenius_norm = np.linalg.norm(result - approxCovarianceMatrix, ord='fro')
#     temp+=frobenius_norm
#     tarr.append(frobenius_norm)

# temp/=100
# print("Average F norm on taking a random permutation of matrix A:",temp)
# plt.plot(np.arange(1,101),tarr)
# plt.show()

# Till now:

# The paper's future work mentions that rows might be available in any order
# We have tried to sample a few rows randomly and show the result
# We also tried to uniformly sample a few rows and showed the result
# We have permuted the rows and shown the result

# Parallelization

f1 = np.linalg.norm(result - approxCovarianceMatrix, ord='fro')

arr = []
arr2 = []
print("Frobenius norm between matrices AT.A and BT.B on performing simple sketching:", f1)

# arr.append(frobenius_norm)

# for j in range(100):
#     A = []
#     for i in range(n):
#         row = dataMaker.makeRow()
#         A.append(row)

#     sketcher = FrequentDirections(d,2*ell)
#     for i in range(0,n//2):
#         sketcher.append(A[i])
#     m1 = sketcher.get()
#     # m1 = dot(sketch.transpose(),sketch)

#     # print(m1.shape)


#     sketcher = FrequentDirections(d,2*ell)
#     for i in range(n//2,n):
#         sketcher.append(A[i])
#     m2 = sketcher.get()
#     # m2 = dot(sketch.transpose(),sketch)
#     # print(m2.shape)

#     m = np.vstack((m1, m2))

#     sketcher = FrequentDirections(d,ell)
#     for i in range(0,m.shape[0]):
#         sketcher.append(m[i])
#     sketch = sketcher.get()
#     M = dot(sketch.transpose(),sketch)
#     # print(M.shape)




#     frobenius_norm = np.linalg.norm(result - M, ord='fro')
#     # print(frobenius_norm)
#     arr.append(frobenius_norm)

#     sketcher = FrequentDirections(d,ell)
#     for i in range(0,n):
#         sketcher.append(A[i])
#     sketch = sketcher.get()

#     approxCovarianceMatrix = dot(sketch.transpose(),sketch)

#     frobenius_norm = np.linalg.norm(result - approxCovarianceMatrix, ord='fro')
#     arr2.append(frobenius_norm)

# plt.plot(np.arange(1,101),arr,label="Parallelized")
# plt.plot(np.arange(1,101),arr2,label="Non-Parallelized")
# plt.xlabel("Experiments")
# plt.ylabel("F-norm")
# plt.legend()
# plt.show()

for j in range(100):
    A = []
    for i in range(n):
        row = dataMaker.makeRow()
        A.append(row)

    sketcher = FrequentDirections(d,3*ell)
    for i in range(0,n//3):
        sketcher.append(A[i])
    m1 = sketcher.get()
    # m1 = dot(sketch.transpose(),sketch)

    print(m1.shape)


    sketcher = FrequentDirections(d,3*ell)
    for i in range(n//3,2*n//3):
        sketcher.append(A[i])
    m2 = sketcher.get()
    # m2 = dot(sketch.transpose(),sketch)
    # print(m2.shape)

    sketcher = FrequentDirections(d,3*ell)
    for i in range(n//3,2*n//3):
        sketcher.append(A[i])
    m3 = sketcher.get()
    # m2 = dot(sketch.transpose(),sketch)
    # print(m2.shape)

    m = np.vstack((m1, m2, m3))

    sketcher = FrequentDirections(d,ell)
    for i in range(0,m.shape[0]):
        sketcher.append(m[i])
    sketch = sketcher.get()
    M = dot(sketch.transpose(),sketch)
    # print(M.shape)




    frobenius_norm = np.linalg.norm(result - M, ord='fro')
    # print(frobenius_norm)
    arr.append(frobenius_norm)

    sketcher = FrequentDirections(d,ell)
    for i in range(0,n):
        sketcher.append(A[i])
    sketch = sketcher.get()

    approxCovarianceMatrix = dot(sketch.transpose(),sketch)

    frobenius_norm = np.linalg.norm(result - approxCovarianceMatrix, ord='fro')
    arr2.append(frobenius_norm)

plt.plot(np.arange(1,101),arr,label="Parallelized")
plt.plot(np.arange(1,101),arr2,label="Non-Parallelized")
plt.xlabel("Experiments")
plt.ylabel("F-norm")
plt.legend()
plt.show()