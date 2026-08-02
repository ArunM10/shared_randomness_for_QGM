#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import time
import torch
# from jax.config import config
# config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import pennylane as qml
from joblib import Parallel, delayed
from jax import jit
from sympy import symbols, sqrt, exp, log, sin, pi, Matrix, expand, eye, trace, collect, Mul, Add
import numba as nb
from numba import jit
from collections import defaultdict
import itertools
from scipy.special import rel_entr


# In[2]:


def sigmoid(x):
        return 1/(1+np.exp(-x))
    
def inv_sigmoid(x):
    return np.log(x/(1-x))


# In[3]:


def plot_prob(p,qubits,depth):
    # cell to find the heatmap
    prob_weights=[]
    for j in range(qubits):
        a=[]
        for i in range(depth):
            a.append((sigmoid((p[:qubits*depth]))).reshape(depth, qubits)[i][j])
        prob_weights.append(a)

    #x_depth = np.linspace(1, 4, 4) ######
    #y_depth = np.linspace(1, 6, 6) ######
    x_depth = -0.5 + np.linspace(1, depth+1, depth+1)  ########
    y_depth = -0.5 + np.linspace(1, qubits+1, qubits+1) ########

    # Generate random weights for each point
    np.random.seed(42)  # Set a seed for reproducibility
    weights = prob_weights
    # Create a meshgrid of the coordinates
    X, Y = np.meshgrid(x_depth, y_depth)

    fig = plt.figure(figsize=(8, 4))

    # Plot the temperature plot
    plt.imshow(weights, cmap='hot', origin='lower', extent=[min(x_depth), max(x_depth), min(y_depth), max(y_depth)], vmin=0, vmax=1)
    plt.colorbar(label='Probabilities')
    plt.ylabel('Qubit')
    plt.xlabel('Depth')
    plt.title('Initial probabilities')
    plt.gca().invert_yaxis()
    plt.show()


# In[5]:


# def target_c_fp(params):  
    
#     p=params[:int(len(params)/2)] # probabilities
#     #print(p)
#     t=params[len(p):] # thetas
    
#     [qml.Hadamard(wires=q) for q in range(qubits)]
#     for l in range(depth): # this is specifically for this problem other "l" should be changed for prob to prob
        
#         for q in range(qubits):
#             if q<qubits-1:
#                 qml.CZ(wires=[q,q+1])
#         qml.CZ(wires=[0,qubits-1])    
#         for q in range(qubits):
#             qml.RZ(t[qubits*l+q],wires=q)
#             qml.Hadamard(wires=q)
        
#             qml.BitFlip((1-sigmoid(p[qubits*l+q]))/2,wires=q)
    
#     return qml.probs(wires=range(qubits))


# # In[6]:


# def model_c_fp(params):  
    
#     p=params[:int(len(params)/2)] # probabilities
#     #print(p)
#     t=params[len(p):] # thetas
    
#     [qml.Hadamard(wires=q) for q in range(qubits)]
#     for l in range(depth): # this is specifically for this problem other "l" should be changed for prob to prob
        
#         for q in range(qubits):
#             if q<qubits-1:
#                 qml.CZ(wires=[q,q+1])
#         qml.CZ(wires=[0,qubits-1])    
#         for q in range(qubits):
#             qml.RZ(t[qubits*l+q],wires=q)
#             qml.Hadamard(wires=q)
            
#             qml.BitFlip((1-sigmoid(p[qubits*l+q]))/2,wires=q)
    
#     return qml.sample()#qml.probs(wires=range(qubits))


# # In[7]:


# def sample_circ(par):
    
#     binary_array = np.array(model_c_fp(par))
#     #print(binary_array)
#     powers_of_two = 2 ** np.arange(binary_array.shape[1])[::-1]
#     decimal_array = np.sum(binary_array * powers_of_two, axis=1)
#     decimal_list = decimal_array.tolist()
    
#     return decimal_list


# In[10]:


def average_with_exclusion(arr, index_to_exclude):
    return sum([x for i, x in enumerate(arr) if i != index_to_exclude]) / (len(arr) - 1)


# In[11]:


##### Rules for propagating byproducts ############

u = symbols('u') # This is a global notation which will be used later 

# This will be a global function 
def t_poly():
    return Matrix([[u**(-1)+u,1],[1,0]])

# This will be a global function 
def t_poly_n(n):
    t=t_poly()
    t=t**n
    return Matrix([[expand(term) for term in row] for row in (t).tolist()])

# This function will represent byproducts in each layer but one thing to must remember** is that the byproducts in the
# last layer should propagate first at the end then the 2nd last and at the end the 1st layer and after all that 
# we have to correct the byproducts at the end
def bp_rep_t(q_idx):
    # q_idx: qubit index in each layer, e.g. [0,1,4]
    arr=[]
    for i in q_idx:
        arr.append(u**i)
    return Matrix([[sum(arr)],[0]])


def transition_act(n,q_idx):
    # pos: position matrix of the byproducts interms of polynomials
    # n: number of transition operators (as layers) acting on the input byproduct layer
    
    r=t_poly_n(n)*bp_rep_t(q_idx)
    return Matrix([[expand(term) for term in row] for row in (r).tolist()])



#### 

def count_distinct_elements(matrix):
    distinct_elements = set()
    for element in matrix:
        if isinstance(element, Add):
            distinct_elements.update(element.args)
        elif element != 0:
            distinct_elements.add(element)
    return distinct_elements


# In[12]:


# Here I want to use the property that for any "n" 2*(u**(n))=0 as the two byprods at same place will be cancelled

def simplify_terms(matrix):
    u = symbols('u')
    
    
    if matrix[0]==1:
        a1=[1]
    elif matrix[0]==0:
        a1=[0]
    else:
        dist_ele_r1=count_distinct_elements(Matrix([matrix[0]]))
    
        if len(dist_ele_r1)==1:
            a1=[]
            t1=matrix[0]
            
            coeff=0
            power=0
            coeff=t1.as_coeff_exponent(u)[0]%2
            power=t1.as_coeff_exponent(u)[1]
            a1.append(coeff*u**(power))
        

        else:
            a1=[]

                # first we do for the 1st row
            
            for t1 in dist_ele_r1:   #matrix[0].args:
                coeff=0
                power=0
                coeff=t1.as_coeff_exponent(u)[0]%2
                power=t1.as_coeff_exponent(u)[1]
                a1.append(coeff*u**(power))
      
    ##############################################
    
    if matrix[1]==1:
        a2=[1]
    elif matrix[1]==0:
        a2=[0]
    else:
        dist_ele_r2=count_distinct_elements(Matrix([matrix[1]]))
        if len(dist_ele_r2)==1:
            a2=[]
            t1=matrix[1]
            
            coeff=0
            power=0
            coeff=t1.as_coeff_exponent(u)[0]%2
            power=t1.as_coeff_exponent(u)[1]
            a2.append(coeff*u**(power))
        

        else:
            a2=[]

                # first we do for the 1st row
            
            for t1 in dist_ele_r2:    #matrix[1].args:
                coeff=0
                power=0
                
                coeff=t1.as_coeff_exponent(u)[0]%2
                power=t1.as_coeff_exponent(u)[1]
                a2.append(coeff*u**(power))
            

    return Matrix([[sum(a1)],[sum(a2)]])


# In[13]:


def next_layer_bp_idx(out_pos):
    
    
    max_idx=qubits
    min_idx=0
    
    # out_pos: position of the byprod after the transition function is applied i.e. transition_act()

    # first we will extract the indices from the out_pos
    x_pos=out_pos[0]
    
    if x_pos==0: #### So if any row has only zero then the default index would be -1 and for that there will be no
                 #### operations in the quantum circuits     
        Xs=[]
        
    elif x_pos==1:
        Xs=[0]
    else:
        dist_ele_r1=count_distinct_elements(Matrix([out_pos[0]]))
        if len(dist_ele_r1)==1:
            x_elements=x_pos
            x_idxs=[x_elements.as_coeff_exponent(u)[1]]
            Xs=[(i % max_idx + max_idx) % max_idx for i in x_idxs]
        else:
            x_elements=x_pos.args
            x_idxs=[i.as_coeff_exponent(u)[1] for i in x_elements]
            Xs=[(i % max_idx + max_idx) % max_idx for i in x_idxs]
            
    
    
    z_pos=out_pos[1]
    if z_pos==0:
        #print('y')
        Zs=[]
        
    elif z_pos==1:
        Zs=[0]
    else:
        dist_ele_r2=count_distinct_elements(Matrix([out_pos[1]]))
        if len(dist_ele_r2)==1:
            z_elements=z_pos# .as_coeff_exponent(u)[1] is used to get the power of 'u'
            z_idxs=[z_elements.as_coeff_exponent(u)[1]]
            Zs=[(i % max_idx + max_idx) % max_idx for i in z_idxs]
        else:
            z_elements=z_pos.args# .as_coeff_exponent(u)[1] is used to get the power of 'u'
            z_idxs=[i.as_coeff_exponent(u)[1] for i in z_elements]
            Zs=[(i % max_idx + max_idx) % max_idx for i in z_idxs]
    
    return Xs,Zs


# In[14]:


def next_layer_bp_idx(out_pos,qubits):
    
    
    max_idx=qubits
    min_idx=0
    
    # out_pos: position of the byprod after the transition function is applied i.e. transition_act()

    # first we will extract the indices from the out_pos
    x_pos=out_pos[0]
    
    if x_pos==0: #### So if any row has only zero then the default index would be -1 and for that there will be no
                 #### operations in the quantum circuits     
        Xs=[]
        
    elif x_pos==1:
        Xs=[0]
    else:
        dist_ele_r1=count_distinct_elements(Matrix([out_pos[0]]))
        if len(dist_ele_r1)==1:
            x_elements=x_pos
            x_idxs=[x_elements.as_coeff_exponent(u)[1]]
            Xs=[(i % max_idx + max_idx) % max_idx for i in x_idxs]
        else:
            x_elements=x_pos.args
            x_idxs=[i.as_coeff_exponent(u)[1] for i in x_elements]
            Xs=[(i % max_idx + max_idx) % max_idx for i in x_idxs]
            
    
    
    z_pos=out_pos[1]
    if z_pos==0:
        #print('y')
        Zs=[]
        
    elif z_pos==1:
        Zs=[0]
    else:
        dist_ele_r2=count_distinct_elements(Matrix([out_pos[1]]))
        if len(dist_ele_r2)==1:
            z_elements=z_pos# .as_coeff_exponent(u)[1] is used to get the power of 'u'
            z_idxs=[z_elements.as_coeff_exponent(u)[1]]
            Zs=[(i % max_idx + max_idx) % max_idx for i in z_idxs]
        else:
            z_elements=z_pos.args# .as_coeff_exponent(u)[1] is used to get the power of 'u'
            z_idxs=[i.as_coeff_exponent(u)[1] for i in z_elements]
            Zs=[(i % max_idx + max_idx) % max_idx for i in z_idxs]
    
    return Xs,Zs


# In[15]:


def cal_bp_only(q_idx,layer,layers,qubits):
    n=layers-layer-1
    
    matrix = transition_act(n,q_idx)

    modified_matrix = simplify_terms(matrix)
    
    return (simplify_terms(bp_from_idx(next_layer_bp_idx((modified_matrix),qubits)))) 


# In[16]:


def bp_from_idx(idxs):
    rows=idxs[0]
    cols=idxs[1]
    
    arr1=[]
    for i in rows:
        arr1.append(u**i)
        
    arr2=[]
    for i in cols:
        arr2.append(u**i)
    return Matrix([[sum(arr1)],[sum(arr2)]])

#bp_from_idx(next_layer_bp_idx((modified_matrix)))


# In[17]:


# This function will calculate the indices for the qubits in each layer which will later be used to caculate 
# the byproducts at the end that needs to be corrected and the indices of layers should be used in reverse order
# i.e.

def layer_qubit_idx(dic,layers,qubits):
        
        indices = {}
        for l in range(layers):
            a = []
            for q in range(qubits):
                if dic[str(l)+' '+str(q)] == 1:
                    a.append(q)
            if a:
                indices[str(l)] = a
        return indices


# In[19]:


def random_choice_2d(probabilities,depth,qubits):
    assert probabilities.shape == (depth, qubits), "Probabilities must have shape (5, 4)"

    result = np.zeros_like(probabilities, dtype=int)
    for index, p in np.ndenumerate(probabilities):
        result[index] = np.random.choice([0, 1], p=[1-sigmoid(p), sigmoid(p)])

    return result
def replace_elements(matrix,depth,qubits):
    assert matrix.shape == (depth, qubits), "Input matrix must have shape (5, 4)"

    replacements = np.where(matrix == 0, np.random.choice([0, 1], size=(depth, qubits), p=[0.5, 0.5]), 0)

    return replacements
#@jit(nopython=True)
def calculate_positions(array):
    positions = []
    for row in array:
        row_positions = np.where(row == 1)[0].tolist()
        positions.append(row_positions)

    return positions


# In[20]:


class VMBQC:
    
    def __init__(self,qubits,layers,N):
        # 
        
        self.qubits=qubits
        self.layers=layers
        self.N=N
     
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def machine_f1(self,P):
        # P = corrections probs that will be used to sample whether we will correct or not 
        
        ### Note that the probs to sample C's is same for all the byproducts, later we can change that and make it
        ### different for different byproducts
    
        qubit_index=np.arange(0,self.qubits)
        layer_index=np.arange(0,self.layers)
        
        x=np.array([0,1]) # 0: We don't correct ; 1: We always correct
        
        
        
        S = np.empty((self.layers * self.qubits,), dtype=int)#[]#np.empty((self.layers , self.qubits))
        C_arr = np.empty((self.layers * self.qubits,), dtype=int)
        #print(len(C_arr))

        # Randomly sample correction values for all qubits and layers in one shot
        
        '''
        for l in layer_index:
            for q in qubit_index:
                
                C_arr[self.qubits*l+q] = np.random.choice(x, 1, p=np.concatenate([np.array([1-self.sigmoid(P[self.qubits*l+q])]), np.array([self.sigmoid(P[self.qubits*l+q])])]))
                
                #C = C_arr[self.qubits*l+q]

                if C_arr[self.qubits*l+q] == 0:
                    # If C == 0 i.e. we don't correct, then byproducts can appear with 1/2 probability
                    s=np.random.choice(x, 1, p=[0.5, 0.5])
                    #print(s)
                    S[self.qubits*l+q] = s
                elif C_arr[self.qubits*l+q] == 1:
                    # If C == 1 i.e. we always correct, then byproducts will not appear at all
                    S[self.qubits*l+q] = 0

        '''
        #s=time.time()
        S=replace_elements(random_choice_2d(P.reshape(self.layers,self.qubits),self.layers,self.qubits),self.layers,self.qubits)

       # e=time.time()
       # print('t2->',e-s)
        return S,C_arr
    
    
    
    
    
    
    
    # rule for correction of the byproducts at the end
    def cal_bp(self,q_idx,layer): # we only require the index of the bp at a specific layer and then we can add them add the end.
        n=self.layers-layer-1

        matrix = transition_act(n,q_idx)

        modified_matrix = simplify_terms(matrix)

        return next_layer_bp_idx(simplify_terms(bp_from_idx(next_layer_bp_idx((modified_matrix)))))
    
    
    
  
    
    
    
    
    def corrected_machine_f2(self,p,t):
        # t = Thetas in the quantum circuits
        
        
        
        b_p=self.machine_f1(p) # These are indices where the byproducts will appear 
        # the first index in b_p states the qubit index and the second one states the layer index
        
        
        
        # Defining each Periodic QCA layer
        def qca_layer(t,l):
            # Layers of CZs
            
            
            # Layers of Rz and H
            for q in range(self.qubits):
                qml.RZ(t[self.qubits*l+q],wires=q)
                
                
            for q in range(self.qubits-1):
                qml.CZ(wires=[q,q+1])
            qml.CZ(wires=[0,self.qubits-1])

            for q in range(self.qubits):
              qml.Hadamard(wires=q)
        
        
        dev=qml.device("lightning.qubit", wires=self.qubits,shots=self.N)
        #@qml.qnode(dev)
        
        def qc(t):
            
            
            [qml.Hadamard(wires=q) for q in range(self.qubits)]
            #s=time.time()
            for l in range(self.layers):
                qca_layer(t,l)
                
                # Introducing byproducts based on previous samples
                for q in range(self.qubits):
                    if b_p[0][l][q]==1:
                        qml.PauliX(wires=q)
            
            
            
            #### Correcting the byproducts
            qubit_idx=calculate_positions(b_p[0])
            
            
            
            index=Matrix([[0],[0]])
            
            
            
            for l in range(self.layers-1,-1,-1):
                
                if len(qubit_idx[l])!=0:
                    
                    index+=(cal_bp_only(qubit_idx[l],l,self.layers,self.qubits))
                    
                else:
                    continue
                    
             
            
            
            
            index=next_layer_bp_idx(simplify_terms(index),self.qubits)
            
            
            
            x_idx=index[0]
                    
            if len(x_idx)!=0:
                #print('y')
                [qml.PauliX(wires=q) for q in x_idx]
            z_idx=index[1]

            if len(z_idx)!=0:
                #print('y')
                [qml.PauliZ(wires=q) for q in z_idx]
            #e=time.time()
            #print('t1->',e-s,'sec')
            
            
            return qml.sample()
        
        
        
        qnode1 = qml.QNode(qc, dev)
        merged_circuit = qml.transforms.cancel_inverses(qc)
        qml.transforms.commutation_dag
        qnode2 = qml.QNode(merged_circuit, dev)
        
        
        return qnode2(t)#,qml.draw_mpl(qnode2)(t),b_p#,b_p#,qml.draw_mpl(qnode1)(t)#,b_p#,qnode2(t),qml.draw_mpl(qnode2)(t),b_p
        


# In[ ]:
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def kernel_exp_torch(s1,s2):
    s1t = torch.tensor(s1, dtype = torch.float64).to(DEVICE)
    s2t = torch.tensor(s2, dtype = torch.float64).to(DEVICE)
    d1 = s1t.size(-1)
    d2 = s2t.size(-1)
    s1t = s1t.reshape([1,-1,1])
    s2t = s2t.reshape([1,-1,1])
    #print(s1t.device)
    
    diffsquared = (torch.cdist(s1t,s2t,p=2.0).to(DEVICE))**2

    sigma_list=[0.25,4]
    exp = 0.0
    for sigma in sigma_list:
        expMatrix = torch.exp(diffsquared/(-2.0*sigma)).to(DEVICE)
        expMatrix = expMatrix/(len(sigma_list)*d1*d2)
        exp = exp + torch.sum(expMatrix).to(DEVICE)
        
    return exp

'''
def mmd_grad_p_new(params):
    p=params[:qubits*depth]
    thetas=params[len(p):]
    grad = torch.zeros(len(p)).to(DEVICE)
    
    
    
    sample_targ=sample_target_function()
    
    for i in range(len(p)):
        
        sample_y = sample_circ(params) ###############  eqv_circ_st(params)
        
        a=p[i].copy()
        #inner_derivative = (sigmoid(torch.tensor(a))**2)*torch.exp(-torch.tensor(a))
        
        p[i]=torch.tensor(1000.0).numpy()
        
        L1=mmd_loss(params)
        
        p[i]=torch.tensor(-1000.0).numpy()
        
        L2=mmd_loss(params)
        
        
        grad[i]=L1-L2
        #grad[i]= grad[i]*inner_derivative
        p[i]=a
        #if i == 0:
        #    print(params[:int(len(params)/2)])

        
    return grad
'''
def mmd_grad_p_new(params):
    p=params[:qubits*depth]
    thetas=params[len(p):]
    grad = torch.zeros(len(p)).to(DEVICE)
    
    
    
    sample_targ=sample_target_function()
    
    for i in range(len(p)):
        
        sample_y = sample_circ(params) ###############  eqv_circ_st(params)
        
        a=p[i].copy()
        inner_derivative = (sigmoid(torch.tensor(a))**2)*torch.exp(-torch.tensor(a))
        
        p[i]=torch.tensor(100.0).numpy()
        
        sample_x_1 = sample_circ(params) ###############  eqv_circ_st(params)
       
        
        p[i]=torch.tensor(-100.0).numpy()
        
        sample_x_0 = sample_circ(params) ###############  eqv_circ_st(params)
        
        
        grad[i]=2*(kernel_exp_torch(sample_y,sample_x_1)-kernel_exp_torch(sample_y,sample_x_0)) - 2*(kernel_exp_torch(sample_targ,sample_x_1)-kernel_exp_torch(sample_targ,sample_x_0))
        grad[i]= grad[i]*inner_derivative
        p[i]=a
        #if i == 0:
        #    print(params[:int(len(params)/2)])

        
    return grad
  
    
    
def mmd_grad_theta(params):
    p=params[:qubits*depth]
    thetas=params[len(p):]
    prob = sample_circ(params)
    grad = torch.zeros(len(thetas)).to(DEVICE)
    
    for i in range(len(thetas)):
        # pi/2 phase
        thetas[i] += np.pi/2.
        prob_pos = sample_circ(params) # p_{theta}^{+}
        # -pi/2 phase
        thetas[i] -= np.pi
        prob_neg = sample_circ(params) # p_{theta}^{-}
        
        t1=kernel_exp_torch(prob, prob_pos)
        
        t2=kernel_exp_torch(prob, prob_neg)

        t3=kernel_exp_torch(sample_target_function(), prob_pos)
        
        t4=kernel_exp_torch(sample_target_function(), prob_neg)

        
        grad_pos = t1 - t2
        grad_neg = t3 - t4
        grad[i] = (grad_pos - grad_neg).detach().cpu()
        
        thetas[i] += np.pi/2.0 #reset to original value
        
    return grad



def mmd_grad(params):
    #probs=params[:int(len(params)/2)]
    #thetas=params[len(probs):]
    grad=torch.cat((mmd_grad_p_new(params),mmd_grad_theta(params)))
    return grad



class GradientDescentOptimizer_mmd_2:  
    def __init__(self, stepsize): # if some value is given here itself then it will become a by default value
        self.stepsize = stepsize

    def step_and_cost(self, objective_fn, *args, grad_fn=None, **kwargs):
        
        g, forward = self.compute_grad(objective_fn, args, kwargs, grad_fn=grad_fn)
        new_args = self.apply_grad(g, args)

        if forward is None:
            forward = objective_fn(*args, **kwargs)

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0], forward
        return new_args, forward


    def step(self, objective_fn, *args, train, grad_fn=None, **kwargs):
        
        p_len=int(len(args[0])/2)
        if train=='p':
            arg1=args[0][:int(len(args[0])/2)]
        else:
            arg1=args[0][int(len(args[0])/2):]
        
        g= self.compute_grad(objective_fn, args, kwargs, grad_fn=grad_fn)
        new_args = self.apply_grad(g, arg1,args,train)

        # unwrap from list if one argument, cleaner return
        if len(new_args) == 1:
            return new_args[0]

        return new_args

    @staticmethod
    def compute_grad(objective_fn, args, kwargs, grad_fn=None):
       
        
        grad = mmd_grad(*args).detach().cpu().numpy()
        

        num_trainable_args = sum(getattr(arg, "requires_grad", False) for arg in args)
        grad = (grad,) if num_trainable_args == 1 else grad
        #print('grads--',grad[0])
        return grad


    def apply_grad(self, grad, arg1,args,train):
        
        #print('arg1--',arg1)#################
        args_new = list(arg1)
        
        if train=='p':

            trained_index = 0
            for index, arg in enumerate(arg1):

                if getattr(arg, "requires_grad", False):
                    args_new[index] = arg - self.stepsize * grad[0][trained_index]

                    trained_index += 1
        else:
            trained_index = p_len ########## need to be generalized
            for index, arg in enumerate(arg1):
                if getattr(arg, "requires_grad", False):
                    args_new[index] = arg - self.stepsize * grad[0][trained_index]

                    trained_index += 1
            
        
        #print('new args-->',args_new[0])#############
        
        if train=='p':
            return np.concatenate([args_new,args[0][len(args_new):]])
        else:
            return np.concatenate([args[0][:len(args_new)],args_new]) # here we can do this only because the they have same size
        
        
from pennylane.numpy import sqrt
class AdagradOptimizer_mmd_2(GradientDescentOptimizer_mmd_2):
    

    def __init__(self, stepsize=0.01, eps=1e-8):
        super().__init__(stepsize)
        self.eps = eps
        self.accumulation = None

    def apply_grad(self, grad, arg1, args, train):
        
        p_len=int(len(args[0])/2)
        args_new = list(arg1)

        if self.accumulation is None:
            self.accumulation = [0.0] * len(arg1)
            
        if train=='p':

            trained_index = 0
            for index, arg in enumerate(arg1):
                if getattr(arg, "requires_grad", False):
                    

                    self._update_accumulation(index, grad[0][trained_index])

                    coeff = self.stepsize / sqrt(self.accumulation[index] + self.eps)
                    args_new[index] = arg - coeff * grad[0][trained_index]

                    trained_index += 1
                    
        else:
            
            trained_index = p_len # need to be generalized
            
            for index, arg in enumerate(arg1):
                if getattr(arg, "requires_grad", False):
                    

                    self._update_accumulation(index, grad[0][trained_index])

                    coeff = self.stepsize / sqrt(self.accumulation[index] + self.eps)
                    args_new[index] = arg - coeff * grad[0][trained_index]

                    trained_index += 1
            
            

        if train=='p':
            return np.concatenate([args_new,args[0][len(args_new):]])
        else:
            return np.concatenate([args[0][:len(args_new)],args_new])


    def _update_accumulation(self, index, grad):
        
        self.accumulation[index] = self.accumulation[index] + grad**2

    def reset(self):
        
        self.accumulation = None


def gen_prob_par(p2,qubits,depth,title):

    # cell to find the heatmap
    prob_weights=[]
    for j in range(qubits):
        a=[]
        for i in range(depth):
            a.append((p2).reshape(depth, qubits)[i][j])
        prob_weights.append(a)
    
    
    x_depth = -0.5 + np.linspace(1, depth+1, depth+1)  ########
    y_depth = -0.5 + np.linspace(1, qubits+1, qubits+1) ########
    
    # Generate random weights for each point
    np.random.seed(42)  # Set a seed for reproducibility
    weights = prob_weights
    # print()
    # Create a meshgrid of the coordinates
    X, Y = np.meshgrid(x_depth, y_depth)
    
    fig = plt.figure(figsize=(8, 4))
    
    # Plot the temperature plot
    plt.imshow(weights, cmap='hot', origin='lower', extent=[min(x_depth), max(x_depth), min(y_depth), max(y_depth)], vmin=0, vmax=1)
    plt.colorbar(label='Probabilities')
    plt.ylabel('Qubit',fontsize=20)
    plt.xlabel('Depth',fontsize=20)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()



from collections import defaultdict


def mul_pauli(p1, p2):
    """
    Multiply Pauli operators ignoring phase.
    """

    if p1 == 'I':
        return p2
    if p2 == 'I':
        return p1
    if p1 == p2:
        return 'I'

    table = {
        ('X', 'Y'): 'Z',
        ('Y', 'X'): 'Z',

        ('X', 'Z'): 'Y',
        ('Z', 'X'): 'Y',

        ('Y', 'Z'): 'X',
        ('Z', 'Y'): 'X',
    }

    return table[(p1, p2)]


def apply_T(paulis, N):
    """
    Apply one CQCA step.

    T(X_i) = X_{i-1} Z_i X_{i+1}
    T(Z_i) = X_i
    """

    out = defaultdict(lambda: 'I')

    for q, P in paulis.items():

        left = (q - 1) % N
        right = (q + 1) % N

        if P == 'Z':

            # Z_i -> X_i
            out[q] = mul_pauli(out[q], 'X')

        elif P == 'X':

            # X_i -> X_{i-1} Z_i X_{i+1}
            out[left] = mul_pauli(out[left], 'X')
            out[q] = mul_pauli(out[q], 'Z')
            out[right] = mul_pauli(out[right], 'X')

        elif P == 'Y':

            # Y = XZ
            # propagate both parts

            # image of X
            out[left] = mul_pauli(out[left], 'X')
            out[q] = mul_pauli(out[q], 'Z')
            out[right] = mul_pauli(out[right], 'X')

            # image of Z
            out[q] = mul_pauli(out[q], 'X')

    return {k: v for k, v in out.items() if v != 'I'}


def propagate_byproducts(z_positions, N, D):
    """
    Initial operator = product of Zs.
    """

    paulis = {q: 'Z' for q in z_positions}

    for _ in range(D):
        paulis = apply_T(paulis, N)

    return paulis

N = 6
D = 5

(propagate_byproducts([0], N, D))