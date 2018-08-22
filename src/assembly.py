import numpy as np


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
        wg *= 0.5(r_max-r_min)
    
    return rg, wg


def cell_average_interaction(r, cells):
    """
    Compute 
    
        I_Ci |r-s|^{-1} ds for all cells in array  
    
    
    Inputs:
    
        r: double, (3,) array point in R^3 
        
        cells: double, (n_cells, 3, 2) array containing the bounds of 
            each cell.
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
        r_min, r_max = cells[:,dimension,:]
        
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
    nz = np.logical_and(a!=0, b!=0, c!=0) 
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


def assemble_spatial_interaction_matrix(level, half_length, n_gauss):
    """
    Assemble matrices used to compute (ij|kl). These terms are used to 
    construct the coulomb and exchange integrals
    
    Inputs:
    
        level: int, approximation level for the Haar wavelets
        
        length: double >0, quantity defining the size of the computational 
            domain, i.e. D = [-half_length, half_length]^3
        
    
    Outputs:
    
        v: double, vector [v1,...,vk] whose terms 
    """
    # =========================================================================
    # Define cells over domain 
    # =========================================================================
    #
    # Define shifts
    # 
    mu_max = 2**level  # Maximum shift
    Mu = np.empty((mu_max**3,3))  # Matrix of 3D shifts
    for (mu,j) in zip(np.mgrid[0:mu_max, 0:mu_max, 0:mu_max], range(dim)):
        Mu[:,j] = mu.ravel() 
    
    #
    # Define cells
    # 
    L = 2*half_length
    lb = L*2**(-level)*(Mu-2**(level-1))
    ub = L*2**(-level)*(1+Mu-2**(level-1))
    cells = np.transpose(np.array([lb,ub]),[1,2,0])
    
    #
    # Generate Quadrature scheme on cell on vertex (-0.5L, -0.5L, -0.5L)
    # 
    rg, wg = gauss_rule(n_gauss, cells[0,:,:])

    n_cells = cell.shape[0]
    v = np.zeros(n_cells)
    for r, w in zip(rg, wg):
        #
        # Compute the integral I_C |s-r| ds for each cell C
        # 
        I = cell_average_interaction(r, cells)
        #
        # Update weighted sum
        # 
        v += w*I