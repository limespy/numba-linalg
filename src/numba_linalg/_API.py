from limedev.CLI import get_main
# ======================================================================
def f() -> None:
    ...
# ======================================================================
main = get_main(__name__)

import numpy as np

def lu_recursive(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError('Matrix must be square')

    def decompose(A):
        n = A.shape[0]
        if n == 1:
            # Base case
            P = np.eye(1)
            L = np.eye(1)
            U = A.copy()
            return P, L, U

        # Partial pivoting
        pivot = np.argmax(abs(A[:, 0]))
        if pivot != 0:
            A[[0, pivot]] = A[[pivot, 0]]
            P = np.eye(n)
            P[[0, pivot]] = P[[pivot, 0]]
        else:
            P = np.eye(n)

        # Partition A
        a11 = A[0, 0]
        a21 = A[1:, 0]
        a12 = A[0, 1:]
        A22 = A[1:, 1:]

        # Compute L and U blocks
        l21 = a21 / a11
        u12 = a12
        S = A22 - np.outer(l21, u12)  # Schur complement

        # Recurse
        P2, L2, U2 = decompose(S)

        # Assemble L and U
        L = np.eye(n)
        L[1:, 0] = l21
        L[1:, 1:] = L2

        U = np.zeros((n, n))
        U[0, :] = np.concatenate([[a11], u12])
        U[1:, 1:] = U2

        # Compose P
        P_full = np.eye(n)
        P_full[1:, 1:] = P2

        P = P @ P_full

        return P, L, U

    return decompose(A)

# Example
if __name__ == '__main__':
    A = np.array([[4, 3], [6, 3]], dtype=float)
    P, L, U = lu_recursive(A)

    print('A:')
    print(A)
    print('\nP:')
    print(P)
    print('\nL:')
    print(L)
    print('\nU:')
    print(U)
    print('\nCheck: P @ A =')
    print(P @ A)
    print('\nL @ U =')
    print(L @ U)
