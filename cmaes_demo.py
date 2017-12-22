'''
demo CMA-ES by Yi-jun Yang
according to the prof. Hansen's homepage, "purecmaes"(MATLAB/Octave Code)
" https://www.lri.fr/~hansen/purecmaes.m "

'''
import time
import numpy as np
from numpy import random
from numpy import linalg as LA

# User defined input parameters (need to be edited)
def benchmark_func(vec):
    dim = len(vec)
    f = 100 * np.sum((vec[0:dim-1]**2 - vec[1:dim])**2) + np.sum((vec[0:dim-1] - 1)**2)
    
    return f

dim = 30
Ubound = 30
Lbound = -30
xmean = Lbound + random.rand(dim) * (Ubound - Lbound)
sigma = 0.3 * (Ubound - Lbound)
max_eva = 10000 * dim

# Strategy parameter setting: Selection
sample_size = int(np.floor(4 + np.floor(3 * np.log(dim)))) # Lambda
mu = sample_size / 2
mu = int(np.floor(mu))
weights = np.log(mu + 0.5) - np.log(np.linspace(1, mu, mu))
weights = weights / np.sum(weights)
mueff = (np.sum(weights))**2 / np.sum(weights**2)

# Strategy parameter setting: Adaptation
cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
cs = (mueff + 2) / (dim + mueff + 5)
c1 = 2 / ((dim+1.3)**2 + mueff)
cmu = min(1-c1, 2*(mueff-2+1/mueff) / ((dim+2)**2+mueff))
damps = 1 + 2*max(0, np.sqrt((mueff-1)/(dim+1))-1) + cs

# Initialize dynamic strategy parameters and constants
pc = np.zeros(dim)
ps = np.zeros(dim)
B = np.eye(dim)
D = np.ones(dim)
C = np.dot(np.dot(B , np.diag(D**2)), B.T)
invsqrtC = np.dot(np.dot(B , np.diag(D**-1)) , B.T) 
eigeneval = 0;                         # track update of B and D
chiN = np.sqrt(dim) * (1-1/(4*dim)+1/(21*dim**2))

start_CPU = time.clock()
#-------------------------Generation Loop-------------------------------------#
counteval = 0
generation = 0
#Result = []
while counteval < max_eva:
    generation += 1
    # Generate and evaluate lambda offspring
    arx = np.ones((dim,sample_size))
    fitness = np.ones(sample_size)
    for k in range(0, sample_size):
        arx[:,k] = xmean + np.ravel(sigma * np.dot(B, D.reshape((dim,1)) * np.random.randn(dim).reshape((dim,1))))
        fitness[k] = benchmark_func(arx[:,k])
        counteval += 1
        
    # Sort by fitness and compute weighted mean into xmean
    order_index = fitness.argsort(kind='quicksort')
    xold = xmean
    mu_index = order_index[0:mu]
    part_arx = arx[:, mu_index]
    xmean = np.dot(part_arx, weights)
    
    # Cumulation: Update evolution paths
    ps = (1-cs) * ps + np.sqrt(cs*(2-cs)*mueff) * np.dot(invsqrtC, (xmean - xold)) / sigma
    if np.sum(ps**2) / (1-(1-cs)**(2*counteval/sample_size))/dim < 2 + 4/(dim+1):
        hsig = 1
    else:
        hsig = 0
    
    pc = (1-cc) * pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (xmean - xold) / sigma
    
    # Adapt covariance matrix C
    artmp = (1/sigma) * (part_arx - np.tile(xold.reshape((dim,1)),[1, mu]))
    C = (1-c1-cmu) * C + c1 * (np.dot(pc.reshape((dim,1)), pc.reshape((dim,1)).T) + (1-hsig) * cc * (2-cc) * C) 
    + cmu * np.dot(np.dot(artmp,np.diag(weights)), artmp.T)
    
    # Adapt step size sigma
    sigma = sigma * np.exp((cs/damps)*(LA.norm(ps)/chiN - 1))
    
    # Update B and D from C
    if counteval - eigeneval > sample_size/(c1+cmu)/dim/10: # to achieve O(N**2)
        eigeneval = counteval
        C = np.triu(C) + np.triu(C, 1).T
        D, B = LA.eig(C)
        D = np.sqrt(D)
        invsqrtC = np.dot(np.dot(B , np.diag(D**-1)) , B.T)
        
    BestValue = benchmark_func(xmean)
    if BestValue < 10**(-10):
        break
#    
#    Result.append(BestValue)
#    print(BestValue,counteval)

end_CPU = time.clock()
print("CPU Time: %f" % (end_CPU - start_CPU))
        
    
                              
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        