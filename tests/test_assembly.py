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




box = np.array([[-2,1], [-1,1], [-4,-2]])
print(I(box))
