import numpy as np

I = np.array([[1,0],[0,1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]]) 
H = np.array([[1,1],[1,-1]])/np.sqrt(2) 
    
def xxz_ham(n,Jxx=[],Jyy=[],Jzz=[],hx=[],hz=[],hy = []):
    
    H = np.zeros([2**n,2**n],complex)
    XX = np.kron(X,X)
    YY = np.kron(Y,Y)
    ZZ = np.kron(Z,Z)
    
    if len(Jxx)!=0:
        for x in range(n-1):
            if Jxx[x]!=0:
                H += Jxx[x]*np.kron(np.eye(2**x),np.kron(XX,np.eye(2**(n-x-2))))        
    if len(Jyy)!=0:
        for x in range(n-1):
            if Jyy[x]!=0:
                H += Jyy[x]*np.kron(np.eye(2**x),np.kron(YY,np.eye(2**(n-x-2))))
    if len(Jzz)!=0:
        for x in range(n-1):
            if Jzz[x]!=0:
                H += Jzz[x]*np.kron(np.eye(2**x),np.kron(ZZ,np.eye(2**(n-x-2))))
    if len(hx)!=0:
        for x in range(n):
            if hx[x]!=0:
                H += hx[x]*np.kron(np.eye(2**x),np.kron(X,np.eye(2**(n-x-1))))   
    if len(hz)!=0:
        for x in range(n):
            if hz[x]!=0:
                H += hz[x]*np.kron(np.eye(2**x),np.kron(Z,np.eye(2**(n-x-1))))
                
    if len(hy)!=0:
        for x in range(n):
            if hy[x]!=0:
                H += hy[x]*np.kron(np.eye(2**x),np.kron(Y,np.eye(2**(n-x-1))))
                
    return H