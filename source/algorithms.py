import numpy as np
import sys
from source.qstate2 import pauli,X,Y,Z
from scipy.linalg import expm,sqrtm
#from source.models import xxz_hamS
from source.qstate2 import mixed_state
#from numpy import logical_not as NOT
from qiskit import QuantumCircuit,Aer,execute,transpile
#from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.monitor import job_monitor
from scipy.linalg import logm
from source.storage_v2 import loaddata
#from scipy.optimize import minimize#,show_options

def find_free_energy(rho,h,beta):
    return beta*np.sum(h.T*rho).real+np.sum(rho.T*logm(rho+1e-12)).real

class processed_data:
    
    def __init__(self,H,beta_values,indx,tp):
        
        # solving eigenproblem
        E,Q = np.linalg.eigh(H)
        
        # computing theoretical prediction from Gibbs distribution
        self.E_gibbs = np.zeros(len(beta_values))
        self.gibbs_distr =  np.zeros(len(beta_values),np.ndarray)
        for bi in range(len(beta_values)):
            self.gibbs_distr[bi] = np.exp(-beta_values[bi]*E)
            self.gibbs_distr[bi] *= 1/np.sum(self.gibbs_distr[bi])
            self.E_gibbs[bi] = np.dot(E,self.gibbs_distr[bi])
        
        # loading data
        rho_avrg_noiseless = loaddata('data/simulations/'+tp+'_noiseless')
        rho_avrg_noisy = loaddata('data/simulations/'+tp+'_noisy')
        rho_optimized = loaddata('data/simulations/'+tp+'_optimized')
          
        # computing overlaps for noiseless data
        rho_ham_basis = np.dot(Q.T.conj(),np.dot(rho_avrg_noiseless[indx],Q))
        self.n_noiseless = np.diag(rho_ham_basis).real
        self.E_noiseless = np.zeros(len(beta_values))
        for bi in range(1,len(beta_values)):
            self.E_noiseless[bi] = np.sum(H.T*rho_avrg_noiseless[bi]).real
        
        # computing overlaps for noisy data
        rho_ham_basis = np.dot(Q.T.conj(),np.dot(rho_avrg_noisy[indx],Q))
        self.n_noisy = np.diag(rho_ham_basis).real
        self.E_noisy = np.zeros(len(beta_values))
        for bi in range(1,len(beta_values)):
            self.E_noisy[bi] = np.sum(H.T*rho_avrg_noisy[bi]).real
        
        # computing for optimized data
        rho_ham_basis = np.dot(Q.T.conj(),np.dot(rho_optimized[indx],Q))
        self.n_optimized = np.diag(rho_ham_basis).real
        self.E_optimized = np.zeros(len(beta_values))
        for bi in range(1,len(beta_values)):
            self.E_optimized[bi] = np.sum(H.T*rho_optimized[bi]).real
            
class algrthm_1:
             
    def Average_Case(self,samples):
    
        for si in range(samples):
            
            sys.stdout.write("\r beta = %f Evaluating sample: %i/%i" % (self.beta,si+1,samples))
            sys.stdout.flush()
            
            self.thetas = np.random.normal(0,1,size = [self.depth,len(self.terms)])
            self.run()
        
            if si==0:
                rho = self.prob*self.state
            if si>0:
                rho += self.prob*self.state
            
        print('')
        return rho/np.trace(rho)
    
    def Optimized_Case(self,samples):
        
        zigzag = np.zeros(samples)
        zigzag[0::5] = 100
        zigzag[1::5] = 100
        zigzag[2::5] = 10
        zigzag[3::5] = 10
        zigzag[4::5] = 1
    
        ths = 0.001*zigzag
        
        #----------------------------------------------------------------------
        
        self.ham()
        rho_gibbs = expm(-self.beta*self.H)
        rho_gibbs = rho_gibbs/np.trace(rho_gibbs)
        free_energy_exact = find_free_energy(rho = rho_gibbs, h = self.H,
                                                              beta = self.beta)
        
        #----------------------------------------------------------------------
        
        free_energy0 = 0
        
        print(r'beta = '+str(self.beta))
        for si in range(samples):
    
            dthetas = ths[si]*np.random.normal(0,1,
                                           size = [self.depth,len(self.terms)])
            self.thetas += dthetas
            self.run()
    
            free_energy = find_free_energy(rho = self.state,h = self.H,
                                                              beta = self.beta)
    
            if free_energy >= free_energy0:
                self.thetas += -dthetas
    
            else:
                free_energy0 = free_energy.copy()
                
            prec = (free_energy0-free_energy_exact)/free_energy_exact
            sys.stdout.write("\r Sample: %i, Free energy: %f (precision = %f)" % (si+1,free_energy0,prec))
            sys.stdout.flush()
        
        return self.state

    def ham(self):
        
        qubit_list = np.unique([item for sublist in self.qubits for item in sublist])
        self.n = len(qubit_list)

        self.H = np.zeros([2**self.n,2**self.n],complex)
        
        for i in range(len(self.terms)):
            
            h = self.terms[i]
            if len(h)==2:
                x = self.qubits[i][0]
                self.H += np.kron(np.eye(2**x),np.kron(h,np.eye(2**(self.n-x-1))))
            if len(h)==4:
                x1,x2 = np.sort(self.qubits[i])
                for a in range(4):
                    for b in range(4):
                        P = np.kron(pauli[a],pauli[b])
                        alpha = np.trace(np.dot(h,P))/4
                        V = np.kron(pauli[b],np.eye(2**(self.n-x2-1)))
                        V = np.kron(np.eye(2**(x2-x1-1)),V)
                        V = np.kron(pauli[a],V)
                        V = np.kron(np.eye(2**x1),V)
                        self.H += alpha*V
        
    def run(self):
        
        self.qubit_list = np.unique([item for sublist in self.qubits for item in sublist])
        self.ancilla_list = np.unique(self.ancilla)
        self.n = len(self.qubit_list)
        self.na = len(self.ancilla_list)
        nf = self.n+self.na
        n_terms = len(self.terms)
        
        # initiate quantum state
        mstate = mixed_state(self.n+self.na)
        mstate.rho = np.diag(np.ones(2**nf))/2**nf
        for xa in self.ancilla_list:
            mstate.reset_thermal(beta='+inf',x=xa)

        # generating square-root operators
        sqrt_terms = np.empty(len(self.terms),np.ndarray)
        for i in range(len(self.terms)):
            h = self.terms[i]
            E0 = np.linalg.eigvalsh(h)[0]
            sqrt_terms[i] = np.kron(sqrtm(-E0*np.eye(len(h))+h),X)
            sqrt_terms[i] *= np.sqrt(self.beta/self.depth)
            
        log_prob = 0
        for di in range(self.depth):
            for i in range(n_terms):
                
                sqrt_h = sqrt_terms[i]
                theta = self.thetas[di,i]
                u = expm(-1j*theta*sqrt_h)
                
                if len(u)==4:
                    x1,x2 = self.qubits[i][0],self.ancilla[i]
                    mstate.apply_2qubit_gate(u,x1,x2)
                    mstate.error(x1,self.p2)
                    mstate.error(x2,self.p2)
                    prob = mstate.postselect_qubit(x2,outcome=0)
                    log_prob += np.log(prob)
                    
                if len(u)==8:
                     x1,x2 = self.qubits[i]
                     x3 = self.ancilla[i]
                     mstate.apply_3qubit_gate(u,x1,x2,x3)
                     mstate.error(x1,self.p3)
                     mstate.error(x2,self.p3)
                     mstate.error(x3,self.p3)
                     prob = mstate.postselect_qubit(x3,outcome=0)
                     log_prob += np.log(prob)
        
        for x in np.flip(self.ancilla_list):
            mstate.discard_qubit(x)
    
        self.state = mstate.rho
        self.prob = np.exp(log_prob)
        
class algrthm_2:
    
    def __init__(self,n,na):
        self.na = na
        self.n = n
        self.depth = 10
        self.beta = 1
        self.gamma = 1
        self.g = 0
        self.lm = 0.1
        
    def Average_Case(self,samples):
    
        for si in range(samples):
            
            sys.stdout.write("\r beta = %f Evaluating sample: %i/%i" % (self.beta,si+1,samples))
            sys.stdout.flush()
            
            self.set_random_schedule()
            self.run()
        
            if si==0:
                rho = self.state
            if si>0:
                rho += self.state
            
        print('')
        return rho/np.trace(rho)
    
    def Optimized_Case(self,samples):
        
        self.set_random_schedule()
        
        sumtime = self.depth/self.gamma
        
        ws = np.zeros(samples)
        ws[2::8] = 0.1
        ws[3::8] = 0.01
        vs = np.zeros(samples)
        vs[4::8] = 0.1
        vs[5::8] = 0.01
        ts = np.zeros(samples)
        ts[6::8] = 0.1
        ts[7::8] = 0.01
        
        #----------------------------------------------------------------------
        
        rho_gibbs = expm(-self.beta*self.H)
        rho_gibbs = rho_gibbs/np.trace(rho_gibbs)
        free_energy_exact = find_free_energy(rho = rho_gibbs, h = self.H,
                                                              beta = self.beta)
        
        #----------------------------------------------------------------------
        
        free_energy0 = 0
        
        omegas = self.omegas.copy()
        X_coupling = self.X_coupling.copy()
        Z_coupling = self.Z_coupling.copy()
        time_schedule = self.time_schedule.copy()
        
        print(r'beta = '+str(self.beta))
        
        for si in range(samples):
    
            self.omegas += ws[si]*np.random.normal(0,1,size = self.na*self.depth).reshape(self.depth,self.na)
            self.X_coupling += vs[si]*np.random.normal(0,1,size = self.na*self.depth).reshape(self.depth,self.na)
            self.Z_coupling += vs[si]*np.random.normal(0,1,size = self.na*self.depth).reshape(self.depth,self.na)
            self.time_schedule = np.abs(self.time_schedule*(1 + ts[si]*np.random.normal(0,1,size = self.depth)))
            self.time_schedule *= sumtime/np.sum(self.time_schedule)
    
            self.run()
    
            free_energy = find_free_energy(rho = self.state,h = self.H,
                                                              beta = self.beta)
    
            if free_energy >= free_energy0:
                self.omegas = omegas.copy()
                self.X_coupling = X_coupling.copy()
                self.Z_coupling = Z_coupling.copy()
                self.time_schedule = time_schedule.copy()
    
            else:
                free_energy0 = free_energy.copy()
                omegas = self.omegas.copy()
                X_coupling = self.X_coupling.copy()
                Z_coupling = self.Z_coupling.copy()
                time_schedule = self.time_schedule.copy()
                
            prec = -(free_energy0-free_energy_exact)/free_energy_exact
            sys.stdout.write("\r Sample: %i, Free energy: %f (precision = %f)" % (si+1,free_energy0,prec))
            sys.stdout.flush()
            
        print()
        
        return self.state
        
    def assign_hamiltonian(self,H):
        self.H = H
        self.E,self.Q = np.linalg.eigh(H)
        
    def set_random_schedule(self):
        s=1
        self.Omega = 1.1*(self.E[-1]-self.E[0])
        self.omegas = self.Omega*(2*np.random.rand(self.depth,self.na)-1)
        self.lm_schedule = self.lm*(1-np.linspace(0,1,self.depth)**(1/s))
        self.X_coupling = np.repeat(1-np.linspace(0,1,self.depth),
                                              self.na).reshape(self.depth,self.na)
        self.Z_coupling = np.repeat(np.zeros(self.depth),
                                               self.na).reshape(self.depth,self.na)
        self.time_schedule = np.random.exponential(1/self.gamma,size = self.depth)

    def run(self):
        
        nf = self.n+self.na
        mstate = mixed_state(self.n+self.na)
        ancillas = np.arange(self.n,nf,1)
        
        for di in range(self.depth):
            
            Hsa  = np.kron(self.H,np.eye(2**self.na))

            T  = self.time_schedule[di]
            
            for xa in range(self.na):
                w = self.omegas[di,xa]
                Hsa += w*np.kron(np.eye(2**(self.n+xa)),
                                np.kron(Z,np.eye(2**(self.na-xa-1))))
                for xs in range(self.n):
                    if self.adj_matrix[xa,xs] == 1:
                        V0 = self.X_coupling[di,xa]*X+self.Z_coupling[di,xa]*Z
                        Vs = np.kron(np.eye(2**xs),
                                     np.kron(V0,np.eye(2**(self.n-xs-1))))
                        Va = np.kron(np.eye(2**xa),
                                     np.kron(X,np.eye(2**(self.na-xa-1))))
                        Hsa += np.kron(Vs,Va)
                        
                mstate.reset_thermal(w*self.beta,ancillas[xa])

            U = expm(-1j*Hsa*T)
            mstate.apply_unitary(U)
            
            for xs in range(self.n):
                mstate.error(xs,min(self.g*T,0.5))
           
        for x in np.flip(ancillas):
            mstate.discard_qubit(x)
    
        self.state = mstate.rho
    
class qiskit_algrthm1():
    
    def __init__(self):
        self.samples_z = np.empty(0,int)
        self.samples_x = np.empty(0,int)
        
    def ham(self):
        self.n = len(self.sys_qubits)
        self.na = len(self.ancillas)
        H = np.zeros([2**self.n,2**self.n],complex)
        qubit_pos = np.zeros(self.n+self.na,int)
        qubit_pos[self.sys_qubits] = np.arange(self.n)
        for ti in range(len(self.terms)):
            if self.terms[ti] == 'z':
                x = qubit_pos[self.subj_qubits[ti][0]]
                H += self.coefs[ti]*np.kron(np.eye(2**x),np.kron(Z,np.eye(2**(self.n-x-1))))
            if self.terms[ti] == 'xx':
                x1,x2 = qubit_pos[np.sort(self.subj_qubits[ti])]
                V = np.kron(X,np.eye(2**(self.n-x2-1)))
                V = np.kron(np.eye(2**(x2-x1-1)),V)
                V = np.kron(X,V)
                V = np.kron(np.eye(2**x1),V)
                H += self.coefs[ti]*V
            if self.terms[ti] == 'yy':
                x1,x2 = qubit_pos[np.sort(self.subj_qubits[ti])]
                V = np.kron(Y,np.eye(2**(self.n-x2-1)))
                V = np.kron(np.eye(2**(x2-x1-1)),V)
                V = np.kron(Y,V)
                V = np.kron(np.eye(2**x1),V)
                H += self.coefs[ti]*V
            if self.terms[ti] == 'zz':
                x1,x2 = qubit_pos[np.sort(self.subj_qubits[ti])]
                V = np.kron(Z,np.eye(2**(self.n-x2-1)))
                V = np.kron(np.eye(2**(x2-x1-1)),V)
                V = np.kron(Z,V)
                V = np.kron(np.eye(2**x1),V)
                H += self.coefs[ti]*V
        self.H = H

    def update_thetas(self):
        n_terms = len(self.terms)
        self.theta_values = np.random.normal(0,1,size = (self.depth,n_terms))        
    
    def run(self):
        
        n_terms = len(self.terms)
        self.theta_values = np.random.normal(0,1,size = (self.depth,n_terms))
        
        qc_list = []
        for si in range(self.circuits):
            sys.stdout.write("\rCompiling circuit: %i" % si)
            sys.stdout.flush()
            #qc_list.append(self.compose_circuit(meas_basis='X'))
            qc_list.append(self.compose_circuit(meas_basis='Z'))
            
        # execute the circuit
        if self.run_type == 'simulator': 
            
            if self.noise:
                backend = self.provider.get_backend(self.device)
                noise_model = NoiseModel.from_backend(backend)
                coupling_map = backend.configuration().coupling_map
                job = execute(qc_list, Aer.get_backend('qasm_simulator'),
                                 coupling_map=coupling_map,
                                 noise_model=noise_model, 
                                 shots = self.shots,
                                 memory=True)
                
            else:
                job = execute(qc_list, Aer.get_backend('qasm_simulator'), 
                                      shots = self.shots, memory=True)

            
        if self.run_type == 'device':
            backend = self.provider.get_backend(self.device)
            qc_list_t = transpile(qc_list, backend, initial_layout = self.layout)
            job = execute(qc_list_t,backend,shots = self.shots,memory=True)
            
        job_monitor(job)
        output = job.result()
        
        num_registers = self.depth*len(self.terms)+self.n
        for k in range(self.circuits):
            circ_output = output.get_memory(k)
            circ_output2 = np.int_([c for c in "".join(circ_output)]).reshape(self.shots,num_registers)
            pslct_output = circ_output2[np.sum(circ_output2[:,self.n:],axis=1)==0][:,:self.n].flatten()
            #if k%2==0:
            #    self.samples_x = np.append(self.samples_x,pslct_output)
            #if k%2==1:
            self.samples_z = np.append(self.samples_z,pslct_output)
        
    # =========================================================================
    #   Manual transpiling of Hadamard and Rx gate using basis gates
    #   for Qiskit.
    # =========================================================================
    
    # Hadamard gate
    def apply_h(self,qc,xq):
        qc.rz(np.pi/2,xq)
        qc.sx(xq)
        qc.rz(np.pi/2,xq)
      
    # Rx gate
    
    def apply_rx(self,qc,angle,xq):
        qc.rz(np.pi/2,xq)
        qc.sx(xq)
        qc.rz(np.pi+angle,xq)
        qc.sx(xq)
        qc.rz(5*np.pi/2,xq)
        
    # =========================================================================
    #   Composition of the circuit
    # =========================================================================
    
    def compose_circuit(self,meas_basis):
        
        self.n  = len(self.sys_qubits)
        self.na = len(self.ancillas)
        nf = self.n + self.na 
        n_terms = len(self.terms)
        qc = QuantumCircuit(nf,n_terms*self.depth+self.n)
        
        for xq in self.sys_qubits:
            qc.reset(xq)
            qc.sx(xq)
            qc.measure(xq,0)

        for xa in self.ancillas:
            qc.reset(xa)

        for di in range(self.depth):
            for ti in range(n_terms):
                
                coef = self.coefs[ti]
                theta = self.theta_values[di,ti]
                f = np.sqrt(2*self.beta*np.abs(coef)/self.depth)
                    
                if self.terms[ti] == 'x':
                    
                    xq = self.subj_qubits[ti]
                    xa = self.subj_ancilla[ti]
                    
                    # Apply thermalizing Z-gate  
                    
                    # xq  -----[H]-----(+)-----[H]----------
                    #                   |
                    # xa  --[X(theta)]--o--[X(theta)]--[M]--
                    
                    self.apply_h(qc,xq)
                    self.apply_rx(qc,np.sign(coef)*f*theta,xa)
                    qc.cx(xa,xq)
                    self.apply_rx(qc,f*theta,xa)
                    self.apply_h(qc,xq)
                    qc.measure(xa,n_terms*di+ti)
                    qc.reset(xa)

                if self.terms[ti] == 'xx':
                    
                    xq1,xq2 = self.subj_qubits[ti]
                    xa = self.subj_ancilla[ti]
                    
                    # Apply thermalizing XX-gate  
                    
                    # xq1 -------------(+)---------------------
                    #                   |
                    # xa  --[X(theta)]--o--o--[X(theta)]--[M]--
                    #                      |
                    # xq2 ----------------(+)------------------
                    
                    self.apply_rx(qc,np.sign(coef)*f*theta,xa)
                    qc.cx(xa,xq1)
                    qc.cx(xa,xq2)
                    self.apply_rx(qc,f*theta,xa)
                    qc.measure(xa,n_terms*di+ti)
                    qc.reset(xa)
                    
                if self.terms[ti] == 'yy':
                    
                    xq1,xq2 = self.subj_qubits[ti]
                    xa = self.subj_ancilla[ti]
                    
                    # Apply thermalizing ZZ-gate  
                    
                    # xq1 ----[S]------(+)--------[S]----------
                    #                   |
                    # xa  --[X(theta)]--o--o--[X(theta)]--[M]--
                    #                      |
                    # xq2 ----[S]---------(+)-----[S]----------
                    
                    qc.rz(np.pi/2,xq1)
                    qc.rz(np.pi/2,xq2)
                    self.apply_rx(qc,np.sign(coef)*f*theta,xa)
                    qc.cx(xa,xq1)
                    qc.cx(xa,xq2)
                    self.apply_rx(qc,f*theta,xa)
                    qc.rz(-np.pi/2,xq1)
                    qc.rz(-np.pi/2,xq2)
                    qc.measure(xa,n_terms*di+ti)
                    qc.reset(xa)
                    
                if self.terms[ti] == 'zz':
                    
                    xq1,xq2 = self.subj_qubits[ti]
                    xa = self.subj_ancilla[ti]
                    
                    # Apply thermalizing ZZ-gate  
                    
                    # xq1 ----[H]------(+)--------[H]----------
                    #                   |
                    # xa  --[X(theta)]--o--o--[X(theta)]--[M]--
                    #                      |
                    # xq2 ----[H]---------(+)-----[H]----------
                    
                    self.apply_h(qc,xq1)
                    self.apply_h(qc,xq2)
                    self.apply_rx(qc,np.sign(coef)*f*theta,xa)
                    qc.cx(xa,xq1)
                    qc.cx(xa,xq2)
                    self.apply_rx(qc,f*theta,xa)
                    self.apply_h(qc,xq1)
                    self.apply_h(qc,xq2)
                    qc.measure(xa,n_terms*di+ti)
                    qc.reset(xa)
                    
        # =====================================================================
        #                      Measuring operators
        # =====================================================================
        
        # rotating to x-basis
        if meas_basis == 'X':
            for xq in self.sys_qubits:
                self.apply_h(qc,xq)
            
        # measure qubits
        s = 0
        for xq in self.sys_qubits:
            qc.measure(xq,n_terms*self.depth+s)
            s += 1
        
        return qc

    
class qiskit_algrthm2:
    
    def __init__(self,n,na):
        self.n = n
        self.na = na
        self.depth = 10
        self.beta = 1
        self.circuits = 100
        self.shots = 1000
        self.samples_z = np.empty(0,int)
        self.samples_x = np.empty(0,int)
        self.noise = False
        
    def set_L_schedule(self,tp = 'linear'):
        
        if tp == 'const':
            self.L_schedule = np.ones(self.depth)
            
        if tp == 'linear':
            self.L_schedule = 1-np.linspace(0,1-1/self.depth,self.depth)
       
    def set_T_schedule(self,tp = 'const'):
        
        if tp == 'const':
            self.T_schedule = np.ones(self.depth)
            
        if tp == 'linear':
            self.T_schedule = np.random.exponential(1/self.gamma,size = self.depth)
            #1/(1-np.linspace(0,1-1/self.depth,self.depth))
            
    def set_V_schedule(self):
        
        self.v_schedule = np.random.normal(0,1,size = (self.depth,self.na,4))
            
    def run(self):
        
        qc_list = []
        for si in range(self.circuits):
            sys.stdout.write("\rCompiling circuit: %i" % si)
            sys.stdout.flush()
            qc_list.append(self.compose_circuit(meas_basis='Z'))
            #qc_list.append(self.compose_circuit(meas_basis='X'))
                     
        # execute the circuit
        if self.run_type == 'simulator':           
            if self.noise:
                backend = self.provider.get_backend(self.device)
                layout = np.hstack((self.sys_qubits+self.ancillas))
                qc_list_t = transpile(qc_list,backend,initial_layout = layout)
                noise_model = NoiseModel.from_backend(backend)
                coupling_map = backend.configuration().coupling_map
                job = execute(qc_list_t, Aer.get_backend('qasm_simulator'),
                                 coupling_map=coupling_map,
                                 noise_model=noise_model, 
                                 shots = self.shots,
                                 memory=True)
                
            else:
                job = execute(qc_list, Aer.get_backend('qasm_simulator'), 
                                      shots = self.shots, memory=True)
    
        if self.run_type == 'device':
            backend = self.provider.get_backend(self.device)
            layout = np.hstack((self.sys_qubits+self.ancillas))      
            qc_list_t = transpile(qc_list,backend,initial_layout = layout)           
            job = execute(qc_list_t,backend,shots = self.shots,memory=True)

        job_monitor(job)
        output = job.result()

        for k in range(self.circuits):
            self.samples_z = np.int_(output.get_memory(k))
            #self.samples_x = np.int_(output.get_memory(2*k+1))
              
    # Hadamard gate
    def apply_h(self,qc,xq):
        qc.rz(np.pi/2,xq)
        qc.sx(xq)
        qc.rz(np.pi/2,xq)
        
    def compose_circuit(self,meas_basis):
        
        sq2 = np.sqrt(2)
        nf = self.n+self.na
        qc = QuantumCircuit(nf,self.n)
        
        for xq in self.sys_qubits:
            qc.reset(xq)
            
        # set system's qubits randomly
        for xq in self.sys_qubits:
            if np.random.choice([True,False]):
                qc.x(xq)
                
        w = np.random.normal(0,1,size=self.na)
        Hs  = np.kron(self.H,np.eye(2**self.na))
        
        for di in range(self.depth):
            
            Ha  = np.zeros([2**nf,2**nf],complex)
            
            Vsa = np.zeros([2**nf,2**nf],complex)
            w = np.random.normal(0,2)
            for xa in range(self.na):
                Ha += w*np.kron(np.eye(2**(self.n+xa)),
                                np.kron(Z,np.eye(2**(self.na-xa-1))))
                for xs in range(self.n):
                    if self.adj_matrix[xa,xs] == 1:
                        a1,a2,a3,a4 = self.v_schedule[di,xa]
                        V0 = [[a1,(a2+1j*a3)/sq2],[(a2-1j*a3)/sq2,a4]]
                        Vs = np.kron(np.eye(2**xs),
                                     np.kron(V0,np.eye(2**(self.n-xs-1))))
                        Va = np.kron(np.eye(2**xa),
                                     np.kron(X,np.eye(2**(self.na-xa-1))))
                        Vsa += np.kron(Vs,Va)
                    
            for xa in self.ancillas:
                qc.reset(xa)
                p0 = np.exp(-self.beta*w)/(2*np.cosh(self.beta*w))
                if np.random.choice([False,True],p = (p0,1-p0)):
                    qc.x(xa)
                   
            lm = self.lm0*self.L_schedule[di]
            T  = self.T0*self.T_schedule[di]
            Hsa = Hs+Ha+lm*Vsa
            U = expm(-1j*Hsa*T)           
            qc.unitary(U,self.sys_qubits+self.ancillas)            
            qc.barrier(np.arange(nf))
            
        # =====================================================================
        #                      Measuring operators
        # =====================================================================
        
        # specify the classical bits for storage
        if self.n>1:
            sys_register = np.arange(self.n)
        if self.n==1:
            sys_register = 0
        
        # rotating to x-basis
        if meas_basis == 'X':
            for xq in self.sys_qubits:
                #qc.h(xq)
                self.apply_h(qc,xq)
            
        # measure qubits
        qc.measure(self.sys_qubits,sys_register)
        
        self.qc_show = qc
        
        return qc