from assembly import *
import matplotlib.pyplot as plt
from scipy.linalg import eig
import numpy as np


# 
assemble_te = False
assemble_tn = True 

#
# Compute the ground state for H2
# 
N = 2  # number of electrons in molecule
halfN = 1
M = 2  # number of nuclei in molecule

# Position of nuclei
R = np.array([[0,0,0],[1.4,0,0]])


#
# Initialize computational mesh
# 
level = 3
half_length = 5
L = 2*half_length
Mu, cells = mesh(half_length, level)
K = len(cells)  # number of basis functions

# 
# Initialize Wave functions
#
A = assemble_laplacian(Mu, L, level)


if assemble_tn:
    # Compute nucleus-electron interaction matrix
    print('Assembling nucleus-electron interaction matrix')
    Tn = assemble_ne_interaction_matrix(R, cells)
    np.save('Tn', Tn)
else:
    Tn = np.load('Tn.npy')

if assemble_te:
    # Electron-electron interaction matrix
    print('Assembling electron-electron interaction matrix')
    Te = assemble_ee_interaction_matrix(cells)
    np.save('Te', Te)
else:
    Te = np.load('Te.npy')
    
#
# Visualize matrices
# 
fig, ax = plt.subplots(nrows=1, ncols=3)

ax[0].imshow(Te)
ax[1].imshow(Tn.todense())

ax[0].set_title('Electron-electron interactions')
ax[1].set_title('Electron-nuclear interactions')


# 
# SCF iteration
#  
print('starting SCF iteration')
C = np.ones((K, halfN))
max_iteration = 30
converged = False
tol = 1e-4
i = 0
e_old = np.zeros(K)
while i < max_iteration:
    #
    # Form Fock Matrix
    # 
    CCT = C.dot(C.T)
    J = 2*np.diag(CCT)*np.diag(Te)
    K = CCT*Te
    F = -0.5*A - Tn + J - K
    
    #
    # Check self-consistency F(C)C = C diag(e)
    # 
    if i > 0:
        error = F.dot(C) - C.dot(np.diag(e))
        error_norm = np.linalg.norm(error)
        
        """
        error = C_old - C
        error_norm = np.linalg.norm(error)
        """
        
        converged = error_norm < tol
        print(error_norm, converged)
    #
    # Compute eigenvalues 
    #         
    C_old = C.copy()
    
    e, C = eig(F)
    
    i += 1
    
ax[2].imshow(F)
ax[2].set_title('Fock Matrix.')
plt.show()