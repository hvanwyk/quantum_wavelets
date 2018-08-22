import numpy as np

def S(a,b,c):
    """
    Compute the triple integral  
    
        I |x|^{-1} dx over R = [0,a]x[0,b]x[0,c], where a,b,c > 0.
        
    Inputs:
    
        a,b,c: np.arrays, integral upper bounds
        
    
    Output:
    
        numpy array
    """
    # 
    # Trivial case
    #
    tol = 1e-12
    
    
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
    
    return I11 + I12 + I21 + I22
      
      
def I(box):
    """
    Integrate even function over a box
    """
    level = 0
    ub = []
    return Ir(box, level, ub)
 
 
def Ir(box, level, ub):
    """
    Recursion function
    """
    val = 0
    if level==3:
        #
        # Base case
        # 
        return S(*ub)
    else:
        #
        # Recursion step
        #
       
        # Update upper bounds
        a, b = box[level,:]
        ub_a, ub_b = list(ub), list(ub)
        ub_b.append(abs(b))
        ub_a.append(abs(a))
        
        # Update integral 
        # I[a,b] = I[b] - I[a]
        val += np.sign(b)*Ir(box, level+1, ub_b)\
             - np.sign(a)*Ir(box, level+1, ub_a)  
        
        return val


dim = 3
L = 10
level = 4
k_max = 2**level
K = np.empty((k_max**3,3))
for (k,j) in zip(np.mgrid[0:k_max, 0:k_max, 0:k_max], range(dim)):
    K[:,j] = k.ravel() 
    
lb = 2*L*2**(-level)*(K-2**(level-1))
ub = 2*L*2**(-level)*(1+K-2**(level-1))

cells = np.transpose(np.array([lb,ub]),[1,2,0])

print(cells[0,:,:])
print(lb.min())
print(ub.min())
print(ub.max())

print(cells.shape)

box = np.array([[-2,1], [-1,1], [4,6]])
print(I(box))


#
# Tensor product quadrature rule on reference cell [-1,1]^3
# 
n_gauss = 7
xg_1d,wg_1d = np.polynomial.legendre.leggauss(n_gauss)
xg_3d = np.empty((n_gauss**dim, dim))
wg_3d = np.ones(n_gauss**dim)
for k,j in zip(np.mgrid[0:n_gauss, 0:n_gauss, 0:n_gauss], range(dim)):
    xg_3d[:,j] = xg_1d[k.ravel()]
    wg_3d *= wg_1d[k.ravel()]

#
# Transform quadrature rule box
# 
n_cells = cells.shape[0]
for ic in range(n_cells):
    yg_3d = np.empty((n_gauss**dim, dim))
    vg_3d = np.ones(n_gauss**dim)
    for id in range(dim):
        y_min = cells[ic,id,0]
        y_max = cells[ic,id,1]
        yg_3d[:,id] = y_min + 0.5*(y_max-y_min)*(xg_3d[:,id]+1)
        vg_3d *= 0.5*(y_max-y_min)
    
    # Check quadrature rule on cell
    if ic==20:
        print(cells[ic,:,:])
        y_min = cells[ic,:,0]
        y_max = cells[ic,:,1]
        #print([(y1,y2) for y1,y2 in zip(y_min, y_max)])
        for id in range(dim):
            #print(yg_3d[:,id].min())
            #print(y_min[id])
            assert np.all(yg_3d[:,id] >= y_min[id]), 'Ohoh'
            assert np.all(yg_3d[:,id] <= y_max[id]), 'ohoh'
            #print(yg_3d.shape) 
        #print(yg_3d.shape)
        #print(vg_3d.shape)
   

"""
Assembly of the matrix (ij|kl)

Must compute I_{C1} I_{C2} |r-s|^{-1} dr ds for all C1, C2 in cells

NOTE: Integral depends only on the relative position of cells. The system
Matrix is therefore Toeplitz (we can represent it by means of one row

quadrule on reference cell
quadrule on c1
for each si in quadrule:
    compute I_{C2} |r-si|^{-1} dr for all C2 in cells:
        transform C2 to C2-si
        split into linear comb of integrals from 0,0,0 to a,b,c (>0)
compute weighted sum

"""
        