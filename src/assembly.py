import numpy as np
import scipy.sparse as sp

def closed_shell_hf(R0, C0, level, half_length, n_gauss=5):
    """
    Solve the closed shell Hartree-Fock equations using a Haar wavelet 
    discretization.
    
    Inputs:
    
        R0: double, (n_nulcei, dim) array of positions of the nuclei
        
        C0: double, (n_cells, n_electrons/2) array of current spatial 
            wave functions
            
        level: int, refinement level in each coordinate
        
        half_length: double >0, 0.5L distance from the center of the 
            hypercube to each of its sides
            
        n_gauss: int, number of Gauss quadrature points in each direction
    
    
    Outputs: 
    
        C: double, (n_cells, n_cells) spatial wave functions 
        
        E: double, (n_cells, ) energies per wave function. 
    """
    # =========================================================================
    # Define Computational Mesh
    # =========================================================================
    Mu, cells = mesh(level, half_length)
    L = 2*half_length
    
    # =========================================================================
    # Precompute A, Te and Tn matrices
    # =========================================================================
    #
    # Laplacian A
    #
    A = assemble_laplacian(Mu, L, level)
    
    #
    # Electron-Electron interactions
    # 
    Te = assemble_ee_interaction_matrix(cells, n_gauss=n_gauss)
    
    #
    # Electron-Nucleus interactions
    #
    Tn = assemble_ne_interaction_matrix(R0, cells)
    
    #
    # SCI procedure
    # 
    for dummy in range(10):
        #
        # Until convergence 
        # 
        
        CCT = C0.dot(C0.T)
        
        # 
        # Coulomb matrix
        # 
        J = 2*np.diag(CCT)*np.diag(Te)
        
        #
        # Exchange matrix
        # 
        K = CCT*Tn
        
        #
        # Fock Matrix
        # 
        F = -0.5*A - Tn + J + K
        
        
    
        
def mesh(half_length, level, dim=3):
    """
    Generate the cells in the computational mesh. 
    
    Inputs: 
        
        half_length: double >0, quantity defining the size of the computational 
            domain, i.e. D = [-half_length, half_length]^3
    
        level: int, refinement level in each direction
        
        dim: int, dimension of mesh (default = 3)
        
    
    Outputs:
    
        Mu: int, (n_cells, 3) array of cell multi-indices 
        
        cells: double, (n_cells, dim, 2) array  of lower and upper bounds
            of each cell in each dimension.
    """
    #
    # Generate cell multi-indices
    # 
    mu_max = 2**level  # Maximum shift
    Mu = np.empty((mu_max**3,3), dtype=np.int)  # Matrix of 3D shifts
    for (mu,j) in zip(np.mgrid[0:mu_max, 0:mu_max, 0:mu_max], range(dim)):
        Mu[:,j] = mu.ravel() 
    
    #
    # Define cells
    # 
    L = 2*half_length
    lb = L*2**(-level)*(Mu-2**(level-1))
    ub = L*2**(-level)*(1+Mu-2**(level-1))
    cells = np.transpose(np.array([lb,ub]),[1,2,0])
    
    return Mu, cells
    
 
        
def assemble_laplacian(Mu, L, level):
    """
    Construct the finite difference Laplacian matrix over 3D hypercube 
    
    
    Inputs:
    
        Mu: int, (K, 3) array of cell indices
    
        L: double >0, length of hypercube in each direction
        
        level: int, refinement level
         
    
    Outputs:
    
        A: double, (K,K) sparse Laplacian matrix 
    """
    n = 2**level  # number of cells in each direction
    h = n/L  # cell width in each direction
     
    # Initialize sparse matrix
    rows = []
    cols = []
    vals = []
    
    #
    # Iterate over Matrix Rows
    #
    K, dim = Mu.shape
    for row in range(K):
        cell_multi_index = Mu[row, :]
        #
        # Consider each dimension
        # 
        for ijk, d in zip(cell_multi_index,range(dim)):
            #
            # Add diagonal term
            # 
            rows.append(row)
            cols.append(row)
            vals.append(-8)
            
            if ijk>0:
                #
                # Cell has 'minus' neighbor 
                #
                cell_nb = cell_multi_index.copy()
                cell_nb[d] = ijk - 1
                
                col = np.ravel_multi_index(tuple(cell_nb), (n,n,n))
                val = 1
                
                # Add entry to system matrix 
                rows.append(row)
                cols.append(col)
                vals.append(val)
                
            if ijk<n-1:
                #
                # Cell has 'plus' neighbor
                # 
                cell_nb = cell_multi_index.copy()
                cell_nb[d] = ijk + 1
                
                col = np.ravel_multi_index(tuple(cell_nb), (n,n,n))
                val = 1
                
                # Add entry to system matrix
                rows.append(row)
                cols.append(col)
                vals.append(val)
    
    # 
    # Construct sparse Laplacian matrix
    # 
    A = 1/h**2*sp.coo_matrix((vals, (rows, cols)), shape=(K,K))        
    return A


def assemble_ee_interaction_matrix(cells, n_gauss=5):
    """
    Compute matrix Te whose ij'th entry is given by
    
    Te_ij = I_Ci I_Cj 1/|r-s| dr ds
    
    
    Input: 
    
        cells: double, (n_cells, dim, 2) array giving lower and upper bounds
            of each cell in each dimension.
            
        n_gauss: int, number of Gauss quadrature points in each direction
         
        
    Output: 
    
        Te: double, full (K,K) matrix whose entries are given above 
        
    """
    n_cells, dim, two = cells.shape
    Te = np.empty((n_cells,n_cells))
    #
    # Iterate over rows
    # 
    for i in range(n_cells):    
        #
        # Generate Quadrature scheme on cell Ci
        # 
        rg, wg = gauss_rule(n_gauss, cells[i,:,:])

        #
        # Compute row of Te matrix
        # 
        Te_row = np.zeros(n_cells)
        for r, w in zip(rg, wg):
            #
            # Compute the integral I_C |s-r| ds for each cell C
            # 
            I = cell_average_interaction(r, cells)
            #
            # Update weighted sum
            # 
            Te_row += w*I
        #
        # Store row in matrix
        # 
        Te[i,:] = Te_row
    #
    # Return full matrix
    # 
    return Te


def assemble_ne_interaction_matrix(R, cells):
    """
    Compute the diagonal matrix, Tn whose ith diagonal entry is 
    
        Tn_i = sum_{Rj} Int_Ci 1/|Rj-r|dr
        
    
    Inputs: 
    
        R: double, (n_nuclei, dim) array of nucleus coordinates
        
        cells: double, (n_cells, dim, 2) array of upper and lower bounds for 
            each cell in each direction. 
            
            
    Outputs:
    
        Tn: double, (n_cells, n_cells) sparse diagonal matrix whose entries
            are given above. 
    """
    n_cells = cells.shape[0]
    Tn_diag = np.zeros(n_cells)
    #
    # Iterate over the nuclei
    # 
    for RA in R:
        #
        # Update diagonal entries
        # 
        Tn_diag += cell_average_interaction(RA, cells)
    #
    # Assemble sparse diagonal matrix
    #   
    Tn = sp.dia_matrix((Tn_diag, 0), shape=(n_cells, n_cells))
    return Tn
    

def gauss_rule(n_gauss, cell):
    """
    Define tensor product quadrature rule (nodes and weights) over reference
    cell. 
    
    Inputs:
    
        n_gauss: int, number of Gauss nodes in each direction
        
        cell: 3x2 matrix, the ith row of which contains the lower and upper 
            bounds of the reference cell in the ith dimension.
            
            cell = [x_min, x_max]
                   [y_min, y_max]
                   [z_min, z_max]
    
    Outputs:
    
        rg: double, (n_gauss**3,3) array of gauss quadrature nodes
        
        wg: double >0, (n_gauss**3) vector of gauss quadrature weights
        
    """
    #
    # Compute one dimensional gauss nodes and weights on [-1,1]
    #
    rg_1d,wg_1d = np.polynomial.legendre.leggauss(n_gauss)
    
    #
    # Form tensor product nodes and weights on reference cell [-1,1]^3 
    #
    dim = 3
    rg = np.empty((n_gauss**dim, dim))
    wg = np.ones(n_gauss**dim)
    for k,i_dim in zip(np.mgrid[0:n_gauss, 0:n_gauss, 0:n_gauss], range(dim)):
        rg[:,i_dim] = rg_1d[k.ravel()]
        wg *= wg_1d[k.ravel()]

    #
    # Transform nodes and weights to physical cell
    # 
    for i in range(dim):
        # Get cell bounds in ith dimension
        r_min = cell[i,0]
        r_max = cell[i,1]
        
        # Modify gauss nodes
        rg[:,i] = r_min + 0.5*(r_max-r_min)*(rg[:,i]+1)
        
        # Modify gauss weights
        wg *= 0.5*(r_max-r_min)
    
    return rg, wg


def cell_average_interaction(r, cells):
    """
    Compute 
    
        I_Ci |r-s|^{-1} ds for all cells Ci in array "cells"  
    
    
    Inputs:
    
        r: double, (3,) array point in R^3 
        
        cells: double, (n_cells, 3, 2) array containing the bounds of 
            each cell.
            
    
    Outputs:
    
        I: double, (n_cells, ) vector of integrals whose ith entry is
            the integral shown above.
    """
    #
    # Translate cells by r
    #
    cells_min_r = np.empty(cells.shape) 
    for ri,i in zip(r, range(3)):
        cells_min_r[:,i,:] = cells[:,i,:] - ri
    
    #
    # Compute integrals recursively 
    # 
    I = cell_average_interaction_recursive(cells_min_r, 0, [])
    return I 


def cell_average_interaction_recursive(cells, dimension, abc):
    """
    Recursive algorithm for computing the integral 
    
                I_Ci |s|^{-1} ds 
        
    over cells of the form Ci = [x_min,x_max]x[y_min,y_max]x[z_min, z_max].
    
    Inputs:
    
        cells: double, (n_cells, n_dim, 2) array of cell bounds
        
        dimension: int, 0,1,2, or 3
        
        abc: double, (dimension,) list of vectors specifying the upper 
            bounds for the region [0,a]x[0,b]x[0,c]
         
    
    Outputs:
    
        I: double, (n_cells, ) vector of integrals whose ith entry is
            the integral shown above.

    """
    n_cells = cells.shape[0]
    I = np.zeros(n_cells)
    if dimension==3:
        #
        # Base case
        # 
        I = S(*abc) 
    else:
        #
        # Recursion step
        #
        
        # Determine lower and upper bounds for given dimension
        r_min = cells[:,dimension,0]
        r_max = cells[:,dimension,1]
        
        # Inherit upper bounds from previous iterated integrals  
        abc_min = list(abc)
        abc_max = list(abc) 
        
        #
        # Decompose dimension'th iterated integral into 
        #    
        #    I[a,b] = I[0,b] - I[0,a]
        #
        # while fixing it so that a,b are positive
        #
        abc_min.append(np.abs(r_min))
        abc_max.append(np.abs(r_max))
        
        # Compute constituent integrals from 0
        I_max = cell_average_interaction_recursive(cells, dimension+1, abc_max)
        I_min = cell_average_interaction_recursive(cells, dimension+1, abc_min)
        
        # Combine Integrals 
        I += np.sign(r_max)*I_max - np.sign(r_min)*I_min
                
    return I 
    
    
def S(a,b,c):
    """
    Compute the triple integral  
    
        I_C |x|^{-1} dx over C = [0,a]x[0,b]x[0,c], where a,b,c > 0.
        
    Inputs:
    
        a,b,c: double >0, (n,) arrays of integral upper bounds
        
    
    Output:
    
        I: double, (n,) array of integrals - one for each cell C 
    """    
    # Initialize integral
    I = np.zeros(a.shape)
    
    # Determine what upper bounds are nonzero
    nz = np.logical_and(np.logical_and(a!=0, b!=0), c!=0) 
    a, b, c = a[nz], b[nz], c[nz]
    
    nrm = np.sqrt(a**2 + b**2 + c**2)
    I11 = - a/2.*np.arctan(b*c/(a*nrm))\
          + 0.5*a*c*np.log((nrm + b)/np.sqrt(a**2 + c**2)) + \
          + 0.5*a*b*np.log((nrm + c)/np.sqrt(a**2 + b**2))
    
    I12 = 0.5*c**2*(np.arcsin(b*c/np.sqrt((a**2+b**2)*(a**2+c**2)))\
          + 0.5*a/c*np.log((b+nrm)/(nrm-b))) -0.5*c**2*np.arctan(b/a)
          
    I21 = - 0.5*b**2*np.arctan(a*c/(b*nrm))\
          + 0.5*a*b*np.log((nrm + c)/np.sqrt(a**2+b**2))\
          + 0.5*b*c*np.log((nrm + a)/np.sqrt(b**2+c**2))
 
    I22 = 0.5*c**2*(np.arcsin(a*c/np.sqrt((a**2+b**2)*(b**2+c**2)))\
          + b/c*np.arctanh(a/nrm)) + 0.5*c**2*(np.arctan(b/a)-0.5*np.pi)
    
    I[nz] = I11 + I12 + I21 + I22

    return I
