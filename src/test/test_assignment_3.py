from src.main.assignment_3 import *

if __name__ == "__main__":
    A_q1 = np.array([
        [3, 1, 1],
        [1, 2, 1],
        [1, 1, 2]
    ], dtype=float)
    b_q1 = np.array([6, 1, 3], dtype=float)
    x_q1 = gaussian_elimination_solve(A_q1, b_q1)
    
    print([int(round(val)) for val in x_q1])  

    A_q2 = np.array([
        [1,  1,  0,  3],
        [2,  1, -1,  1],
        [3, -1, -1,  2],
        [-1, 2,  3, -1]
    ], dtype=float)
    L_q2, U_q2 = lu_factorization(A_q2)
    detA_q2 = determinant_cofactor(A_q2)

    print(detA_q2)
    print(L_q2)
    print(U_q2)

    A_q3 = np.array([
        [9,  0,  5,  2,  1],
        [3,  9,  1,  2,  1],
        [0,  1,  7,  2,  3],
        [4,  2,  3, 12,  2],
        [3,  2,  4,  0,  8]
    ], dtype=float)
    diag_dom_result = is_diagonally_dominant(A_q3)
    print("Is diagonally dominant?", diag_dom_result)

    A_q4 = np.array([
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ], dtype=float)
    pos_def_result = is_positive_definite(A_q4)
    print("Is positive definite?", pos_def_result)

