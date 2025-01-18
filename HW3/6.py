import numpy as np
from scipy.linalg import qr

# Part 1: QR Decomposition of an Orthogonal Matrix
np.random.seed(42)  # For reproducibility
A = np.random.rand(6, 6)  # Generate a random 6x6 matrix
Q, R = qr(A)  # Perform QR decomposition

# Verify R is identity when recomputing QR on Q
Q_check, R_check = qr(Q)
print("Part 1: Orthogonal Matrix QR Decomposition")
print("R (should be identity):\n", np.round(R_check, 2))

# Part 2: Modify Norms of Columns
norms = np.arange(10, 16)  # Norms: 10, 11, 12, 13, 14, 15
U = Q * norms
Q_mod, R_mod = qr(U)

print("\nPart 2: Modified Norms")
print("Diagonal of R (should match norms 10-15):\n", np.round(np.diag(R_mod), 2))

# Verify Q.T * Q
orthogonality_check = np.round(Q_mod.T @ Q_mod, 2)
print("Q.T @ Q (should be identity):\n", orthogonality_check)

# Part 3: Break Orthogonality
U[0, 3] = 0  # Break orthogonality
Q_break, R_break = qr(U)

print("\nPart 3: Broken Orthogonality")
print("R (no longer clean upper triangular):\n", np.round(R_break, 2))
