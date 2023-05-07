import numpy as np
import sys
from source.qstate2 import pauli,X,Y,Z
from scipy.linalg import expm,sqrtm
from source.qstate2 import mixed_state
from qiskit import QuantumCircuit,Aer,execute,transpile
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.monitor import job_monitor
from scipy.linalg import logm
from source.storage_v2 import loaddata

def find_free_energy(rho,h,beta):

    # free energy of state 'rho' in respect to Hamiltonian 'h'
    
    return beta*np.sum(h.T*rho).real+np.sum(rho.T*logm(rho+1e-12)).real        

# ================================================================================
#                 Noisy simulations of the algorithms
# ================================================================================

class algrthm_1:

    # Universal Algorithm

    def Average_Case(self,samples):

        # Calculate the average output of a quantum circuit with a 
        # Gaussian distribution of angles.
    
        # Loop through all the samples
        for si in range(samples):
            
            # Display progress in the console
            sys.stdout.write("\r beta = %f Evaluating sample: %i/%i" % (self.beta,si+1,samples))
            sys.stdout.flush()
            
            # generating angles for Gaussian distribution
            self.thetas = np.random.normal(0,1,size = [self.depth,len(self.terms)])

            # run the circuit
            self.run()
        
            # add state multiplied by acceptance probability
            if si == 0:
                rho = self.prob*self.state
            if si > 0:
                rho += self.prob*self.state
            
        # Print a newline character for console output formatting
        print('')

        # return normalized density matrix
        return rho/np.trace(rho)
    
    def Optimized_Case(self,samples):

        # Find the circuit with values of angles optimized by noise 
        
        #----------------------------------------------------------------------
        # Part 1: Setup the optimization schedule

        # set the relative size of steps in stochastic optimization
        zigzag = np.zeros(samples)
        zigzag[0::5] = 100
        zigzag[1::5] = 100
        zigzag[2::5] = 10
        zigzag[3::5] = 10
        zigzag[4::5] = 1
    
        # set the absolute value of steps
        ths = 0.001*zigzag
        
        #----------------------------------------------------------------------
        # Part 2: Find the traget value of free energy
        
        # generate the Hamiltonian matrix
        self.ham()

        # generate Gibbs state
        rho_gibbs = expm(-self.beta*self.H)
        rho_gibbs = rho_gibbs/np.trace(rho_gibbs)

        # evaluate the taraget value of free energy
        free_energy_exact = find_free_energy(rho = rho_gibbs, h = self.H,
                                                              beta = self.beta)
        
        #----------------------------------------------------------------------
        # Step 3. Run the optimization
        
        # set the initial avlaue of free energy
        free_energy0 = 0
        
        # display the value of inverse temperature
        print(r'beta = '+str(self.beta))

        # Loop through all the samples
        for si in range(samples):
    
            # perturb the angle values
            dthetas = ths[si]*np.random.normal(0,1,
                                           size = [self.depth,len(self.terms)])
            self.thetas += dthetas

            # run the circuit
            self.run()
    
            # evaluate the free energy
            free_energy = find_free_energy(rho = self.state,h = self.H,
                                                              beta = self.beta)
    
            # reject the move if the free energy is larger
            if free_energy >= free_energy0:
                self.thetas += -dthetas
    
            # otherwise accept the move
            else:
                free_energy0 = free_energy.copy()
                
            # evaluate the resulting precision in free energy
            prec = (free_energy0-free_energy_exact)/free_energy_exact

            # Display progress in the console
            sys.stdout.write("\r Sample: %i, Free energy: %f (precision = %f)" % (si+1,free_energy0,prec))
            sys.stdout.flush()
        
        return self.state

    def ham(self):

        #---------------------------
        # Evaluate the Hamiltonian
        #---------------------------
        
        # form the list of unique qubits
        qubit_list = np.unique([item for sublist in self.qubits for item in sublist])

        # find the number of qubits
        self.n = len(qubit_list)

        # initialize Hamiltonian matrix
        self.H = np.zeros([2**self.n,2**self.n],complex)
        
        # Loop through all local terms
        for i in range(len(self.terms)):
            
            # choose the term
            h = self.terms[i]

            # adding single-qubit term
            if len(h)==2:
                x = self.qubits[i][0]
                self.H += np.kron(np.eye(2**x),np.kron(h,np.eye(2**(self.n-x-1))))

            # adding two-qubit term
            if len(h)==4:

                # ordering qubit positions
                x1,x2 = np.sort(self.qubits[i])

                # decomposing on Pauli terms and adding to Hamiltonian
                for a in range(4):
                    for b in range(4):

                        # define local Pauli operator
                        P = np.kron(pauli[a],pauli[b])

                        # define Pauli coefficient
                        alpha = np.trace(np.dot(h,P))/4

                        # generating Pauli term
                        V = np.kron(pauli[b],np.eye(2**(self.n-x2-1)))
                        V = np.kron(np.eye(2**(x2-x1-1)),V)
                        V = np.kron(pauli[a],V)
                        V = np.kron(np.eye(2**x1),V)

                        # adding Pauli term with coefficient
                        self.H += alpha*V
        
    def run(self):

        #---------------------------
        # Run the circuit
        #---------------------------
        
        # read the qubit list
        self.qubit_list = np.unique([item for sublist in self.qubits for item in sublist])

        # read the ancilla list
        self.ancilla_list = np.unique(self.ancilla)

        # define the number of qubits
        self.n = len(self.qubit_list)

        # define the number of ancilla
        self.na = len(self.ancilla_list)

        # total number of qubits
        nf = self.n + self.na

        # total number of Hamiltonian terms
        n_terms = len(self.terms)
        
        # initiate quantum state in inifnite-temperature state
        mstate = mixed_state(self.n+self.na)
        mstate.rho = np.diag(np.ones(2**nf))/2**nf
        for xa in self.ancilla_list:
            mstate.reset_thermal(beta='+inf',x=xa)

        # generating square-root operators corresponding to each Hamiltonian term
        sqrt_terms = np.empty(len(self.terms),np.ndarray)
        for i in range(len(self.terms)):
            h = self.terms[i]
            E0 = np.linalg.eigvalsh(h)[0]
            sqrt_terms[i] = np.kron(sqrtm(-E0*np.eye(len(h))+h),X)
            sqrt_terms[i] *= np.sqrt(self.beta/self.depth)
            
        # initialize logarithm of accaptance probability
        log_prob = 0

        # Loop through all circuit layers
        for di in range(self.depth):

            # Loop through all gates in the layer
            for i in range(n_terms):
                
                # take square-root term
                sqrt_h = sqrt_terms[i]

                # take the corresponding angle
                theta = self.thetas[di,i]

                # generate gate unitary
                u = expm(-1j*theta*sqrt_h)
                
                # apply noisy 2-qubit gate (for 1-qubit term)
                if len(u)==4:

                    # set the qubit coordinates
                    x1,x2 = self.qubits[i][0],self.ancilla[i]

                    # apply the gate
                    mstate.apply_2qubit_gate(u,x1,x2)

                    # apply the errors
                    mstate.error(x1,self.p2)
                    mstate.error(x2,self.p2)

                    # find the probability of acceptance
                    prob = mstate.postselect_qubit(x2,outcome = 0)

                    # add its logarithm to the total log-probability
                    log_prob += np.log(prob)
                    
                # apply noisy 3-qubit gate (for 2-qubit term)
                if len(u)==8:
                     
                     # set the qubit coordinates
                     x1,x2 = self.qubits[i]
                     x3 = self.ancilla[i]

                     # apply the gate
                     mstate.apply_3qubit_gate(u,x1,x2,x3)

                     # apply the errors
                     mstate.error(x1,self.p3)
                     mstate.error(x2,self.p3)
                     mstate.error(x3,self.p3)

                     # find the probability of acceptance
                     prob = mstate.postselect_qubit(x3,outcome=0)

                     # add its logarithm to the total log-probability
                     log_prob += np.log(prob)
        
        # discard ancilla
        for x in np.flip(self.ancilla_list):
            mstate.discard_qubit(x)
    
        # write the result as output of the algorithm
        self.state = mstate.rho

        # evaluate the acceptance probability
        self.prob = np.exp(log_prob)
        
class algrthm_2:

     # Ergodic Algorithm
    
    def __init__(self,n,na):
        self.na = na      # number of ancilla
        self.n = n        # number of qubits
        self.depth = 10   # nymber of reset cycles (default)
        self.beta = 1     # inverse temperature (default)
        self.gamma = 1    # inverse cycle time (default)
        self.g = 0        # noise rate (default)
        self.lm = 0.1     # system-ancilla coupling (default)
        
    def Average_Case(self,samples):

        # Calculate the average output of a quantum circuit with a 
        # Gaussian distribution of angles.
    
        # Loop through all the samples
        for si in range(samples):
            
            # Display progress in the console
            sys.stdout.write("\r beta = %f Evaluating sample: %i/%i" % (self.beta,si+1,samples))
            sys.stdout.flush()
            
            # set the schedule with random evolution times
            self.set_random_schedule()

            # run the circuit
            self.run()
        
            # add state to average
            if si==0:
                rho = self.state
            if si>0:
                rho += self.state

        # Print a newline character for console output formatting
        print('')

        # return normalized density matrix
        return rho/np.trace(rho)
    
    def Optimized_Case(self,samples):

        # Find the circuit with parameters optimized for noise 
        
        #----------------------------------------------------------------------
        # Part 1: Setup the optimization schedule
        
        # generate the random schedule/parameters
        self.set_random_schedule()
        
        # evaluate the total evolution time
        sumtime = self.depth/self.gamma
        
        # set the size of the step for ancilla frequencies
        ws = np.zeros(samples)
        ws[2::8] = 0.1
        ws[3::8] = 0.01

        # set the size of the steps for V_{km} terms
        vs = np.zeros(samples)
        vs[4::8] = 0.1
        vs[5::8] = 0.01

        # set the steps for evolution time
        ts = np.zeros(samples)
        ts[6::8] = 0.1
        ts[7::8] = 0.01

        #----------------------------------------------------------------------
        # Part 2: Find the traget value of free energy
        
        # generate Gibbs state
        rho_gibbs = expm(-self.beta*self.H)

        # find the target value of free energy
        rho_gibbs = rho_gibbs/np.trace(rho_gibbs)
        free_energy_exact = find_free_energy(rho = rho_gibbs, h = self.H,
                                                              beta = self.beta)
        
        #----------------------------------------------------------------------
        # Step 3. Run the optimization of the evolution

        # set the initial value of free energy
        free_energy0 = 0
        
        # set the initial values of parameters
        omegas = self.omegas.copy()                # ancilla frequencies
        X_coupling = self.X_coupling.copy()        # X-pauli coefficients for operators V_{km}
        Z_coupling = self.Z_coupling.copy()        # Z-pauli coefficients for operators V_{km}
        time_schedule = self.time_schedule.copy()  # evolution times
        
        # Display the value of inverse temperature we are working with
        print(r'beta = '+str(self.beta))
        
        # Lopp through optimization steps
        for si in range(samples):
    
            # perturb the values of parameters with corresponding amplitudes
            self.omegas += ws[si]*np.random.normal(0,1,size = self.na*self.depth).reshape(self.depth,self.na)
            self.X_coupling += vs[si]*np.random.normal(0,1,size = self.na*self.depth).reshape(self.depth,self.na)
            self.Z_coupling += vs[si]*np.random.normal(0,1,size = self.na*self.depth).reshape(self.depth,self.na)
            self.time_schedule = np.abs(self.time_schedule*(1 + ts[si]*np.random.normal(0,1,size = self.depth)))
            self.time_schedule *= sumtime/np.sum(self.time_schedule)
    
            # run the circuit
            self.run()
    
            # evaluate the free energy
            free_energy = find_free_energy(rho = self.state,h = self.H,
                                                              beta = self.beta)
    
            # reject the move if the free energy is larger than current value
            if free_energy >= free_energy0:
                self.omegas = omegas.copy()
                self.X_coupling = X_coupling.copy()
                self.Z_coupling = Z_coupling.copy()
                self.time_schedule = time_schedule.copy()
    
            # otherwise, accept the changes
            else:
                free_energy0 = free_energy.copy()
                omegas = self.omegas.copy()
                X_coupling = self.X_coupling.copy()
                Z_coupling = self.Z_coupling.copy()
                time_schedule = self.time_schedule.copy()
                
            # compute the precision
            prec = -(free_energy0-free_energy_exact)/free_energy_exact

            # Display the updates in the console
            sys.stdout.write("\r Sample: %i, Free energy: %f (precision = %f)" % (si+1,free_energy0,prec))
            sys.stdout.flush()
            
        # Print a newline character for console output formatting
        print('')

        # return the output
        return self.state
        
    def assign_hamiltonian(self,H):
        # Take the Hamiltonian as input, diagobalizes it and stores the result
        self.H = H
        self.E,self.Q = np.linalg.eigh(H)
        
    def set_random_schedule(self):

        #-------------------------------
        # Generate random schedule
        #-------------------------------

        # set the range of ancilla frequencies slightly larger than 
        # the spectral norm of the Hamiltonian
        self.Omega = 1.1*(self.E[-1]-self.E[0])

        # generate random values of ancilla frequencies in the range above
        self.omegas = self.Omega*(2*np.random.rand(self.depth,self.na)-1)

        # set the couplings decreasing with depth
        s = 1
        self.lm_schedule = self.lm*(1-np.linspace(0,1,self.depth)**(1/s))

        # set random values of $V_{km}$ Pauli coefficients
        self.X_coupling = np.repeat(1-np.linspace(0,1,self.depth),
                                              self.na).reshape(self.depth,self.na)
        self.Z_coupling = np.repeat(np.zeros(self.depth),
                                               self.na).reshape(self.depth,self.na)
        
        # generate times from Poissonian distribution
        self.time_schedule = np.random.exponential(1/self.gamma,size = self.depth)

    def run(self):

        #-------------------------------
        # Running the circuit
        #-------------------------------
        
        # get the total number of qubits
        nf = self.n+self.na

        # initiate the system in infinite-temperature state
        mstate = mixed_state(self.n+self.na)

        # attach the indices to ancilla in range [n,...,nf-1]
        ancillas = np.arange(self.n,nf,1)
        
        # Loop through the cycles of the evolution
        for di in range(self.depth):
            
            # initial the system-ancilla Hamiltonian adding system Hamiltonian
            Hsa  = np.kron(self.H,np.eye(2**self.na))

            # get the evolution time
            T  = self.time_schedule[di]
            
            # Loop though adding ancilla Hamiltonians and the couplings between 
            # ancilla and the system
            for xa in range(self.na):

                # get the ancilla qubit frequency
                w = self.omegas[di,xa]
                
                # add ancilla Hamiltonian
                Hsa += w * np.kron(np.eye(2**(self.n+xa)),
                                np.kron(Z,np.eye(2**(self.na-xa-1))))
                
                # add coupling according to the system-ancilla adjacency matrix
                for xs in range(self.n):

                    if self.adj_matrix[xa,xs] == 1:

                        # generate coupling term $V_{km}$
                        V0 = self.X_coupling[di,xa] * X + self.Z_coupling[di,xa] * Z

                        # convert it into multi-qubit operator acting on system qubits 
                        Vs = np.kron(np.eye(2**xs),
                                     np.kron(V0,np.eye(2**(self.n-xs-1))))
                        
                        # generate 'X' multi-qubit operator acting on ancilla
                        Va = np.kron(np.eye(2**xa),
                                     np.kron(X,np.eye(2**(self.na-xa-1))))
                        
                        # add the term to the Hamiltonian
                        Hsa += np.kron(Vs,Va)
                        
                # reset ancilla before the cycle
                mstate.reset_thermal(w*self.beta,ancillas[xa])

            # generate the cycle unitary
            U = expm(-1j*Hsa*T)

            # apply the unitary
            mstate.apply_unitary(U)
            
            # apply the errors
            for xs in range(self.n):
                mstate.error(xs,min(self.g*T,0.5))
           
        # discard ancilla qubits
        for x in np.flip(ancillas):
            mstate.discard_qubit(x)
    
        # write the circuit output as the algorithm's output
        self.state = mstate.rho

# ================================================================================
#           Implementation of the algorithms using IBM Qiskit 
# ================================================================================
    
class qiskit_algrthm1():

    # Qiskit Universal Algorithm
    
    def __init__(self):
        self.samples_z = np.empty(0,int)
        self.samples_x = np.empty(0,int)
        
    def ham(self):

        #-------------------------------
        # Generate the Hamiltonian
        #-------------------------------

        # get the number of system qubits
        self.n = len(self.sys_qubits)

        # get the number of acnilla qubits
        self.na = len(self.ancillas)

        # initialize the Hamiltonian as zero matrix
        H = np.zeros([2**self.n,2**self.n],complex)

        # generate the list of qubit positions
        qubit_pos = np.zeros(self.n + self.na,int)
        qubit_pos[self.sys_qubits] = np.arange(self.n)

        # Loop through the Hamiltonian terms 
        for ti in range(len(self.terms)):

            # adding Pauli-Z
            if self.terms[ti] == 'z':
                x = qubit_pos[self.subj_qubits[ti][0]]
                H += self.coefs[ti]*np.kron(np.eye(2**x),np.kron(Z,np.eye(2**(self.n-x-1))))

            # adding Pauli-XX
            if self.terms[ti] == 'xx':
                x1,x2 = qubit_pos[np.sort(self.subj_qubits[ti])]
                V = np.kron(X,np.eye(2**(self.n-x2-1)))
                V = np.kron(np.eye(2**(x2-x1-1)),V)
                V = np.kron(X,V)
                V = np.kron(np.eye(2**x1),V)
                H += self.coefs[ti]*V

            # adding Pauli-YY
            if self.terms[ti] == 'yy':
                x1,x2 = qubit_pos[np.sort(self.subj_qubits[ti])]
                V = np.kron(Y,np.eye(2**(self.n-x2-1)))
                V = np.kron(np.eye(2**(x2-x1-1)),V)
                V = np.kron(Y,V)
                V = np.kron(np.eye(2**x1),V)
                H += self.coefs[ti]*V

            # adding Pauli-ZZ
            if self.terms[ti] == 'zz':
                x1,x2 = qubit_pos[np.sort(self.subj_qubits[ti])]
                V = np.kron(Z,np.eye(2**(self.n-x2-1)))
                V = np.kron(np.eye(2**(x2-x1-1)),V)
                V = np.kron(Z,V)
                V = np.kron(np.eye(2**x1),V)
                H += self.coefs[ti]*V

        self.H = H

    def update_thetas(self):

        #-----------------------------------------------------------------
        # Generate list of angles from Gaussian distribution (to remove?)
        #-----------------------------------------------------------------

        # get the number of terms = number of angles
        n_terms = len(self.terms)

        # generate the list of angles
        self.theta_values = np.random.normal(0,1,size = (self.depth,n_terms))        
    
    def run(self):

        #---------------------------
        # Run the circuits
        #---------------------------
        
        # get the number of terms = number of angles
        n_terms = len(self.terms)

        # generate the list of angles
        self.theta_values = np.random.normal(0,1,size = (self.depth,n_terms))
        
        # generate a list of Qiskit quantum circuits
        qc_list = []
        for si in range(self.circuits):
            sys.stdout.write("\rCompiling circuit: %i" % si)
            sys.stdout.flush()
            qc_list.append(self.compose_circuit(meas_basis='Z'))
            
        # simulate the execution of the circuit on a local computer
        if self.run_type == 'simulator': 
            
            # noisy simulation
            if self.noise:

                # get device-sepcific noise model
                backend = self.provider.get_backend(self.device)
                noise_model = NoiseModel.from_backend(backend)

                # get device-specific coupling map
                coupling_map = backend.configuration().coupling_map

                # execute the job
                job = execute(qc_list, Aer.get_backend('qasm_simulator'),
                                 coupling_map = coupling_map,
                                 noise_model = noise_model, 
                                 shots = self.shots,
                                 memory=True)
                
            # noiseless simulation
            else:

                # execute the job
                job = execute(qc_list, Aer.get_backend('qasm_simulator'), 
                                      shots = self.shots, memory=True)
            
        # run the code on actual device
        if self.run_type == 'device':

            # get the device's backend
            backend = self.provider.get_backend(self.device)

            # transpile the circuit using native gates (mostly nominal)
            qc_list_t = transpile(qc_list, backend, initial_layout = self.layout)

            # execute the job
            job = execute(qc_list_t,backend,shots = self.shots,memory=True)
            
        # monitor the job running
        job_monitor(job)

        # collect and store the shot data
        output = job.result()
        num_registers = self.depth*len(self.terms)+self.n
        for k in range(self.circuits):
            circ_output = output.get_memory(k)
            circ_output2 = np.int_([c for c in "".join(circ_output)]).reshape(self.shots,num_registers)
            pslct_output = circ_output2[np.sum(circ_output2[:,self.n:],axis=1)==0][:,:self.n].flatten()
            self.samples_z = np.append(self.samples_z,pslct_output)
    
    def apply_h(self,qc,xq):

        # Manual transpiling of Hadamard gate

        qc.rz(np.pi/2,xq)
        qc.sx(xq)
        qc.rz(np.pi/2,xq)
      
    def apply_rx(self,qc,angle,xq):

        # Manual transpiling of Rx gate

        qc.rz(np.pi/2,xq)
        qc.sx(xq)
        qc.rz(np.pi+angle,xq)
        qc.sx(xq)
        qc.rz(5*np.pi/2,xq)
    
    def compose_circuit(self,meas_basis):

        # ----------------------------------------
        #  The circuit composition
        # ----------------------------------------
        
        # get the number of qubits
        self.n  = len(self.sys_qubits)

        # get the number of ancilla
        self.na = len(self.ancillas)

        # get the total number of qubits
        nf = self.n + self.na 

        # get the total number of terms = number of angles
        n_terms = len(self.terms)

        # initial quantum circuit
        qc = QuantumCircuit(nf,n_terms*self.depth+self.n)
        
        # generate fully mixed state by applying Hadamard and measurement
        for xq in self.sys_qubits:
            qc.reset(xq)
            qc.sx(xq)
            qc.measure(xq,0)

        # resetting the ancilla
        for xa in self.ancillas:
            qc.reset(xa)

        # Loop throug the circuit cycles
        for di in range(self.depth):

            # Loop through the gates
            for ti in range(n_terms):
                
                # generate the angle for Rx gate
                coef = self.coefs[ti]
                theta = self.theta_values[di,ti]
                f = np.sqrt(2*self.beta*np.abs(coef)/self.depth)
                    
                # 2-qubit gate for Z-term in the Hamiltonian
                if self.terms[ti] == 'z':
                    
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

                # 3-qubit gate for XX-term in the Hamiltonian
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
                    
                # 3-qubit gate for YY-term in the Hamiltonian
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
                    
                # 3-qubit gate for ZZ-term in the Hamiltonian
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
                    
        # ---------------------
        #  Measuring operators
        # ---------------------
        
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

    # Qiskit Ergodic Algorithm
    
    def __init__(self,n,na):
        self.n = n                        # number of qubits
        self.na = na                      # number of ancilla qubits
        self.depth = 10                   # total number of cycles
        self.beta = 1                     # inverse temperature
        self.circuits = 100               # number of circuits
        self.shots = 1000                 # number of shots
        self.samples_z = np.empty(0,int)  # samples in z-basis
        self.samples_x = np.empty(0,int)  # samples in x-basis
        self.noise = False                # adding noise in simulation
        
    def set_L_schedule(self,tp = 'linear'):
        
        # Setting the algorithm's coupling schedule

        # option 1: coupling remains the same for all cycles
        if tp == 'const':
            self.L_schedule = np.ones(self.depth)
            
        # option 2: coupling decreases linearly with number of cycles
        if tp == 'linear':
            self.L_schedule = 1-np.linspace(0,1-1/self.depth,self.depth)
       
    def set_T_schedule(self,tp = 'const'):

        # Setting the algorithm's evolution times
        
        # option 1: times are the same for all cycles
        if tp == 'const':
            self.T_schedule = np.ones(self.depth)
            
         # option 2: times are sampled from Poisson distribution
        if tp == 'linear':
            self.T_schedule = np.random.exponential(1/self.gamma,size = self.depth)
            
    def set_V_schedule(self):
        
        # Set coupling operator $V_{km}# Pauli coefficient to be Gaussian random variables
        self.v_schedule = np.random.normal(0,1,size = (self.depth,self.na,4))
            
    def run(self):

        #---------------------------
        # Run the circuits
        #---------------------------
        
        # generate the list of Qiskit circuits
        qc_list = []
        for si in range(self.circuits):
            sys.stdout.write("\rCompiling circuit: %i" % si)
            sys.stdout.flush()
            qc_list.append(self.compose_circuit(meas_basis='Z'))
                     
        # simulate the execution of the circuit on a local computer
        if self.run_type == 'simulator':    

            # noisy simulation       
            if self.noise:

                # get the device-specific noise model
                backend = self.provider.get_backend(self.device)
                noise_model = NoiseModel.from_backend(backend)

                # get the device-specific qubits layout
                coupling_map = backend.configuration().coupling_map

                # transpile the circuit
                layout = np.hstack((self.sys_qubits+self.ancillas)) 
                qc_list_t = transpile(qc_list,backend,initial_layout = layout)
                
                # run the job
                job = execute(qc_list_t, Aer.get_backend('qasm_simulator'),
                                 coupling_map = coupling_map,
                                 noise_model = noise_model, 
                                 shots = self.shots,
                                 memory=True)
                
            # noiseless simulation
            else:
                job = execute(qc_list, Aer.get_backend('qasm_simulator'), 
                                      shots = self.shots, memory=True)
    
        # run the experiment on actual IBM device
        if self.run_type == 'device':

            # get the backend
            backend = self.provider.get_backend(self.device)

            # transpile the circuit
            layout = np.hstack((self.sys_qubits+self.ancillas))      
            qc_list_t = transpile(qc_list,backend,initial_layout = layout)  

            # run the job         
            job = execute(qc_list_t,backend,shots = self.shots,memory=True)

        # Display the job progress
        job_monitor(job)

        # store the obtained shot samples
        output = job.result()
        for k in range(self.circuits):
            self.samples_z = np.int_(output.get_memory(k))
              
    def apply_h(self,qc,xq):

        # manually transpiled Hadamard gate

        qc.rz(np.pi/2,xq)
        qc.sx(xq)
        qc.rz(np.pi/2,xq)
        
    def compose_circuit(self,meas_basis):
        
        sq2 = np.sqrt(2)

        # get the total number of qubits
        nf = self.n+self.na

        # initial Qiskit circuit
        qc = QuantumCircuit(nf,self.n)
        
        # reset system qubit
        for xq in self.sys_qubits:
            qc.reset(xq)
            
        # set system's qubits in random state
        for xq in self.sys_qubits:
            if np.random.choice([True,False]):
                qc.x(xq)
                
        # set the ancilla frequencies randomly
        w = np.random.normal(0,1,size=self.na)
        Hs  = np.kron(self.H,np.eye(2**self.na))
        
        for di in range(self.depth):
            
            # generate Hamiltonian
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
                    
            # reset the ancilla qubits in the thermal state
            for xa in self.ancillas:
                qc.reset(xa)
                p0 = np.exp(-self.beta*w)/(2*np.cosh(self.beta*w))
                if np.random.choice([False,True],p = (p0,1-p0)):
                    qc.x(xa)
                   
            # get the cycle coupling
            lm = self.lm0*self.L_schedule[di]

            #get the cycle time
            T  = self.T0*self.T_schedule[di]

            # get the system-ancilla Hamiltonian
            Hsa = Hs+Ha+lm*Vsa

            # generate the system-ancilla cycle unitary
            U = expm(-1j*Hsa*T)   

            # apply the unitary        
            qc.unitary(U,self.sys_qubits+self.ancillas)     

            # add the barrier between the cycles       
            qc.barrier(np.arange(nf))
            
        # ------------------------------------------
        #           Measuring operators
        # ------------------------------------------
        
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