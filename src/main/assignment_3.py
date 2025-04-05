import numpy as np

def gaussian_elimination_solve(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = A.shape[0]
    
    # Forward elimination
    for i in range(n):
        pivot = A[i, i]
        for j in range(i+1, n):
            if A[j, i] != 0:
                m = A[j, i]/pivot
                A[j, i:] = A[j, i:] - m*A[i, i:]
                b[j] = b[j] - m*b[i]
    
    # Backward substitution
    x = np.zeros(n)
    for i in reversed(range(n)):
        sum_ax = 0
        for j in range(i+1, n):
            sum_ax += A[i, j]*x[j]
        x[i] = (b[i] - sum_ax) / A[i, i]
    
    return x

def lu_factorization(A):
    U = A.astype(float).copy()
    n = U.shape[0]

    L = np.eye(n, dtype=float)

    for i in range(n):
        pivot = U[i, i]

        for j in range(i+1, n):
            m = U[j, i] / pivot
            L[j, i] = m
            U[j, i:] = U[j, i:] - m * U[i, i:]

    return L, U

def determinant_cofactor(matrix):
    A = np.array(matrix, dtype=float)

    if A.shape == (1, 1):
        return A[0, 0]

    if A.shape == (2, 2):
        return A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]

    det_val = 0.0
    n = A.shape[0]
    for col in range(n):
        submatrix = np.delete(np.delete(A, 0, axis=0), col, axis=1)
        cofactor = ((-1) ** col) * A[0, col]
        det_val += cofactor * determinant_cofactor(submatrix)

    return det_val

def is_diagonally_dominant(mat):
    n = mat.shape[0]
    for i in range(n):
        diag = abs(mat[i, i])
        off_sum = np.sum(np.abs(mat[i])) - diag
        if diag < off_sum:
            return False
    return True

def is_positive_definite(A, tol=1e-12):
    A = np.array(A, dtype=float)
    n, m = A.shape

    if n != m:
        return False

    if not np.allclose(A, A.T, atol=tol):
        return False

    U = A.copy()

    for i in range(n):
        pivot = U[i, i]
        if pivot <= 0:
            # If the pivot is non-positive, not positive definite (interchange needed)
            return False

        for j in range(i+1, n):
            m_factor = U[j, i] / pivot
            U[j, i:] = U[j, i:] - m_factor * U[i, i:]

    return True

