import numpy as np
import matplotlib.pyplot as plt
from numpy import logical_and as AND
from numpy import logical_not as NOT
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

rho0 = np.array([[1,0],[0,0]])
 
I = np.array([[1,0],[0,1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]]) 
H = np.array([[1,1],[1,-1]])/np.sqrt(2) 
S = np.array([[1,0],[0,1j]])
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
CZ = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
XX = np.kron(X,X)
YY = np.kron(Y,Y)
ZZ = np.kron(Z,Z)
pauli = [I,X,Y,Z]

def logical_zeros(n,x):
    # group sizes
    d1,d2 = 2**(n-x-1),2**x
    logic0  = np.tile(np.repeat([True,False],d1),d2)
    return logic0

class circuit_1d():
    
    def __init__(self,size,depth):
        self.size = size
        self.depth = depth
        self.circuit = np.zeros([depth,size],int)
        self.gate_alphabet = []
        self.gate_size = []
        self.gate_color = []
        
    def show(self,ax):
        #gate_color = ['yellow','blue','orange','pink','red']
        for x in range(self.size):
            ax.plot([-1,2*len(self.circuit)-1],[x,x],c='k',zorder=0)
        patches = []
        col = []
        for d in range(len(self.circuit)):
            for x in range(len(self.circuit.T)):
                if self.circuit[d,x]>0:
                    sq=int(self.gate_size[self.circuit[d,x]-1]==2)
                    x0,y0,xs,ys = 2*d,x+0.5-0.5*sq,1,1.75-1*sq
                    rect = mpatches.Rectangle([x0-xs/2,y0-ys/2], 
                                              xs, ys, edgecolor = 'k')
                    patches.append(rect)
                    col.append(self.gate_color[self.circuit[d,x]-1])
        collection = PatchCollection(patches, cmap=plt.cm.hsv,facecolor=col,
                                     edgecolor = 'k',zorder=1)
        ax.add_collection(collection)
        ax.set_ylim(-1,len(self.circuit.T)+1)
        ax.set_axis_off()
    
class mixed_state():
    
    def __init__(self,n):
        self.size = n
        self.rho = np.zeros([2**n,2**n],complex)
        self.rho[0,0] = 1
    
    def apply_unitary(self,U):
        self.rho = np.dot(U,np.dot(self.rho,U.T.conj()))
        
    def apply_noisy_evolution(self,H,time,p):
        self.rho = self.rho
        
    def error(self,x,p):
        
        # step 1: defining logical basis of two qubits A and B
        logic0 = logical_zeros(self.size,x)
        logic1 = NOT(logic0)
        # step2: performing the unitary transform
        
        rhoX = 0*self.rho
        rhoX[logic0] = self.rho[logic1]
        rhoX[logic1] = self.rho[logic0]
        rhoX_new = 1*rhoX
        rhoX_new.T[logic0] = rhoX.T[logic1]
        rhoX_new.T[logic1] = rhoX.T[logic0]
        
        rhoY = 0*self.rho
        rhoY[logic0] = -1j*self.rho[logic1]
        rhoY[logic1] = 1j*self.rho[logic0]
        rhoY_new = 1*rhoY
        rhoY_new.T[logic0] = 1j*rhoY.T[logic1]
        rhoY_new.T[logic1] = -1j*rhoY.T[logic0]
        
        rhoZ = 0*self.rho
        rhoZ[logic0] = self.rho[logic0]
        rhoZ[logic1] = -self.rho[logic1]
        rhoZ_new = 1*rhoZ
        rhoZ_new.T[logic0] = rhoZ.T[logic0]
        rhoZ_new.T[logic1] = -rhoZ.T[logic1]
        
        self.rho = (1-p)*self.rho + p/3*rhoX_new\
                                  + p/3*rhoY_new\
                                  + p/3*rhoZ_new
    
    def apply_1qubit_gate(self,u,x):
        
        uc = u.conj()
        # step 1: defining logical basis of two qubits A and B
        logic0 = logical_zeros(self.size,x)
        logic1 = NOT(logic0)
        # step2: performing the unitary transform
        
        rho_new = 0*self.rho
        rho_new[logic0] = u[0,0]*self.rho[logic0]+u[0,1]*self.rho[logic1]
        rho_new[logic1] = u[1,0]*self.rho[logic0]+u[1,1]*self.rho[logic1]
        self.rho = rho_new.copy()
        
        rho_new = 0*self.rho
        rho_new.T[logic0] = uc[0,0]*self.rho.T[logic0]+uc[0,1]*self.rho.T[logic1]
        rho_new.T[logic1] = uc[1,0]*self.rho.T[logic0]+uc[1,1]*self.rho.T[logic1]
        self.rho = rho_new.copy()
        
    def apply_2qubit_gate(self,u,x1,x2):
        
        uc = u.conj()
        
        # step 1: defining logical basis of two qubits A and B
        logic0A = logical_zeros(self.size,x1)
        logic0B = logical_zeros(self.size,x2)
        logic1A,logic1B = NOT(logic0A),NOT(logic0B)
        logic00 = AND(logic0A,logic0B)
        logic01 = AND(logic0A,logic1B)
        logic10 = AND(logic1A,logic0B)
        logic11 = AND(logic1A,logic1B)
        # step2: performing the unitary transform
        rho_new = 0j*self.rho
        rho_new[logic00] = u[0,0]*self.rho[logic00]+u[0,1]*self.rho[logic01]+\
                           u[0,2]*self.rho[logic10]+u[0,3]*self.rho[logic11]
        rho_new[logic01] = u[1,0]*self.rho[logic00]+u[1,1]*self.rho[logic01]+\
                           u[1,2]*self.rho[logic10]+u[1,3]*self.rho[logic11]
        rho_new[logic10] = u[2,0]*self.rho[logic00]+u[2,1]*self.rho[logic01]+\
                           u[2,2]*self.rho[logic10]+u[2,3]*self.rho[logic11]
        rho_new[logic11] = u[3,0]*self.rho[logic00]+u[3,1]*self.rho[logic01]+\
                           u[3,2]*self.rho[logic10]+u[3,3]*self.rho[logic11]
        self.rho = rho_new.copy() 
        
        rho_new = 0j*self.rho
        rho_new.T[logic00] = uc[0,0]*self.rho.T[logic00]+uc[0,1]*self.rho.T[logic01]+\
                             uc[0,2]*self.rho.T[logic10]+uc[0,3]*self.rho.T[logic11]
        rho_new.T[logic01] = uc[1,0]*self.rho.T[logic00]+uc[1,1]*self.rho.T[logic01]+\
                             uc[1,2]*self.rho.T[logic10]+uc[1,3]*self.rho.T[logic11]
        rho_new.T[logic10] = uc[2,0]*self.rho.T[logic00]+uc[2,1]*self.rho.T[logic01]+\
                             uc[2,2]*self.rho.T[logic10]+uc[2,3]*self.rho.T[logic11]
        rho_new.T[logic11] = uc[3,0]*self.rho.T[logic00]+uc[3,1]*self.rho.T[logic01]+\
                             uc[3,2]*self.rho.T[logic10]+uc[3,3]*self.rho.T[logic11]
        self.rho = rho_new.copy() 
            
    def apply_3qubit_gate(self,u,x1,x2,x3):
        
        uc = u.conj()
        
        # step 1: defining logical basis of two qubits A and B
        logic0A = logical_zeros(self.size,x1)
        logic0B = logical_zeros(self.size,x2)
        logic0C = logical_zeros(self.size,x3)
        logic1A,logic1B,logic1C = NOT(logic0A),NOT(logic0B),NOT(logic0C)
        logic000 = AND(AND(logic0A,logic0B),logic0C)
        logic001 = AND(AND(logic0A,logic0B),logic1C)
        logic010 = AND(AND(logic0A,logic1B),logic0C)
        logic011 = AND(AND(logic0A,logic1B),logic1C)
        logic100 = AND(AND(logic1A,logic0B),logic0C)
        logic101 = AND(AND(logic1A,logic0B),logic1C)
        logic110 = AND(AND(logic1A,logic1B),logic0C)
        logic111 = AND(AND(logic1A,logic1B),logic1C)
        
        # step2: performing the unitary transform
        rho_new = 0j*self.rho
        rho_new[logic000] = u[0,0]*self.rho[logic000]+u[0,1]*self.rho[logic001]+\
                            u[0,2]*self.rho[logic010]+u[0,3]*self.rho[logic011]+\
                            u[0,4]*self.rho[logic100]+u[0,5]*self.rho[logic101]+\
                            u[0,6]*self.rho[logic110]+u[0,7]*self.rho[logic111]
        rho_new[logic001] = u[1,0]*self.rho[logic000]+u[1,1]*self.rho[logic001]+\
                            u[1,2]*self.rho[logic010]+u[1,3]*self.rho[logic011]+\
                            u[1,4]*self.rho[logic100]+u[1,5]*self.rho[logic101]+\
                            u[1,6]*self.rho[logic110]+u[1,7]*self.rho[logic111]
        rho_new[logic010] = u[2,0]*self.rho[logic000]+u[2,1]*self.rho[logic001]+\
                            u[2,2]*self.rho[logic010]+u[2,3]*self.rho[logic011]+\
                            u[2,4]*self.rho[logic100]+u[2,5]*self.rho[logic101]+\
                            u[2,6]*self.rho[logic110]+u[2,7]*self.rho[logic111]
        rho_new[logic011] = u[3,0]*self.rho[logic000]+u[3,1]*self.rho[logic001]+\
                            u[3,2]*self.rho[logic010]+u[3,3]*self.rho[logic011]+\
                            u[3,4]*self.rho[logic100]+u[3,5]*self.rho[logic101]+\
                            u[3,6]*self.rho[logic110]+u[3,7]*self.rho[logic111]
        rho_new[logic100] = u[4,0]*self.rho[logic000]+u[4,1]*self.rho[logic001]+\
                            u[4,2]*self.rho[logic010]+u[4,3]*self.rho[logic011]+\
                            u[4,4]*self.rho[logic100]+u[4,5]*self.rho[logic101]+\
                            u[4,6]*self.rho[logic110]+u[4,7]*self.rho[logic111]
        rho_new[logic101] = u[5,0]*self.rho[logic000]+u[5,1]*self.rho[logic001]+\
                            u[5,2]*self.rho[logic010]+u[5,3]*self.rho[logic011]+\
                            u[5,4]*self.rho[logic100]+u[5,5]*self.rho[logic101]+\
                            u[5,6]*self.rho[logic110]+u[5,7]*self.rho[logic111]
        rho_new[logic110] = u[6,0]*self.rho[logic000]+u[6,1]*self.rho[logic001]+\
                            u[6,2]*self.rho[logic010]+u[6,3]*self.rho[logic011]+\
                            u[6,4]*self.rho[logic100]+u[6,5]*self.rho[logic101]+\
                            u[6,6]*self.rho[logic110]+u[6,7]*self.rho[logic111]
        rho_new[logic111] = u[7,0]*self.rho[logic000]+u[7,1]*self.rho[logic001]+\
                            u[7,2]*self.rho[logic010]+u[7,3]*self.rho[logic011]+\
                            u[7,4]*self.rho[logic100]+u[7,5]*self.rho[logic101]+\
                            u[7,6]*self.rho[logic110]+u[7,7]*self.rho[logic111]
        self.rho = rho_new.copy()
        
        rho_new = 0j*self.rho
        rho_new.T[logic000] = uc[0,0]*self.rho.T[logic000]+uc[0,1]*self.rho.T[logic001]+\
                              uc[0,2]*self.rho.T[logic010]+uc[0,3]*self.rho.T[logic011]+\
                              uc[0,4]*self.rho.T[logic100]+uc[0,5]*self.rho.T[logic101]+\
                              uc[0,6]*self.rho.T[logic110]+uc[0,7]*self.rho.T[logic111]
        rho_new.T[logic001] = uc[1,0]*self.rho.T[logic000]+uc[1,1]*self.rho.T[logic001]+\
                              uc[1,2]*self.rho.T[logic010]+uc[1,3]*self.rho.T[logic011]+\
                              uc[1,4]*self.rho.T[logic100]+uc[1,5]*self.rho.T[logic101]+\
                              uc[1,6]*self.rho.T[logic110]+uc[1,7]*self.rho.T[logic111]
        rho_new.T[logic010] = uc[2,0]*self.rho.T[logic000]+uc[2,1]*self.rho.T[logic001]+\
                              uc[2,2]*self.rho.T[logic010]+uc[2,3]*self.rho.T[logic011]+\
                              uc[2,4]*self.rho.T[logic100]+uc[2,5]*self.rho.T[logic101]+\
                              uc[2,6]*self.rho.T[logic110]+uc[2,7]*self.rho.T[logic111]
        rho_new.T[logic011] = uc[3,0]*self.rho.T[logic000]+uc[3,1]*self.rho.T[logic001]+\
                              uc[3,2]*self.rho.T[logic010]+uc[3,3]*self.rho.T[logic011]+\
                              uc[3,4]*self.rho.T[logic100]+uc[3,5]*self.rho.T[logic101]+\
                              uc[3,6]*self.rho.T[logic110]+uc[3,7]*self.rho.T[logic111]
        rho_new.T[logic100] = uc[4,0]*self.rho.T[logic000]+uc[4,1]*self.rho.T[logic001]+\
                              uc[4,2]*self.rho.T[logic010]+uc[4,3]*self.rho.T[logic011]+\
                              uc[4,4]*self.rho.T[logic100]+uc[4,5]*self.rho.T[logic101]+\
                              uc[4,6]*self.rho.T[logic110]+uc[4,7]*self.rho.T[logic111]
        rho_new.T[logic101] = uc[5,0]*self.rho.T[logic000]+uc[5,1]*self.rho.T[logic001]+\
                              uc[5,2]*self.rho.T[logic010]+uc[5,3]*self.rho.T[logic011]+\
                              uc[5,4]*self.rho.T[logic100]+uc[5,5]*self.rho.T[logic101]+\
                              uc[5,6]*self.rho.T[logic110]+uc[5,7]*self.rho.T[logic111]
        rho_new.T[logic110] = uc[6,0]*self.rho.T[logic000]+uc[6,1]*self.rho.T[logic001]+\
                              uc[6,2]*self.rho.T[logic010]+uc[6,3]*self.rho.T[logic011]+\
                              uc[6,4]*self.rho.T[logic100]+uc[6,5]*self.rho.T[logic101]+\
                              uc[6,6]*self.rho.T[logic110]+uc[6,7]*self.rho.T[logic111]
        rho_new.T[logic111] = uc[7,0]*self.rho.T[logic000]+uc[7,1]*self.rho.T[logic001]+\
                              uc[7,2]*self.rho.T[logic010]+uc[7,3]*self.rho.T[logic011]+\
                              uc[7,4]*self.rho.T[logic100]+uc[7,5]*self.rho.T[logic101]+\
                              uc[7,6]*self.rho.T[logic110]+uc[7,7]*self.rho.T[logic111]
                            
        self.rho = rho_new.copy()
        
                
    def postselect_qubit(self,x,outcome):
        # step 1: defining logical basis of two qubits A and B
        logic0 = logical_zeros(self.size,x)
        logic1 = NOT(logic0)
        # step 2: apply measurement
        p0 = np.round(np.sum(np.diag(self.rho)[logic0]).real,12)
        rho = np.zeros([2**self.size,2**self.size],complex)
        prob = 0
        if outcome == 0 and p0!=0:
            logic00 = np.outer(logic0,logic0)
            rho[logic00] = self.rho[logic00]
            self.rho = rho.copy()
            prob = p0
        if outcome == 1 and p0!=1:
            logic11 = np.outer(logic1,logic1)
            rho[logic11] = self.rho[logic11]
            self.rho = rho.copy()
            prob = 1-p0
        self.rho = self.rho/np.trace(self.rho)
        return prob
    
    def reset_thermal(self,beta,x,p=1):
        
        if beta!='+inf' and beta!='-inf':
            f0,f1 = np.exp(-beta)/(2*np.cosh(beta)),np.exp(+beta)/(2*np.cosh(beta))
        if beta=='+inf':
            f0,f1 = 1,0
        if beta=='-inf':
            f0,f1 = 0,1

        logic0 = logical_zeros(self.size,x)
        logic1 = NOT(logic0)
        logic0_mtrx = np.outer(logic0,logic0)
        logic1_mtrx = np.outer(logic1,logic1)
        
        rho_subs = self.rho[logic0_mtrx]+self.rho[logic1_mtrx] 
        self.rho = (1-p)*self.rho
        self.rho[logic0_mtrx] += p*f0*rho_subs
        self.rho[logic1_mtrx] += p*f1*rho_subs
        
    def discard_qubit(self,x):
        logic0 = logical_zeros(self.size,x)
        logic1 = NOT(logic0)
        rho_new = np.zeros([2**(self.size-1),2**(self.size-1)],complex)
        logic00 = np.outer(logic0,logic0)
        logic11 = np.outer(logic1,logic1)
        D = self.size-1
        rho_new = (self.rho[logic00]+self.rho[logic11]).reshape(2**D,2**D)
        self.rho = rho_new.copy()
        self.size += -1