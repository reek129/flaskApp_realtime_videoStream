#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pennylane as qml
n_qubits = 4

# In[3]:


# independentlayer
def H_layer(nqubits):
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


# In[4]:


def RY_layer(w):
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)
        


# In[5]:


def RX_layer(w):
    for idx, element in enumerate(w):
        qml.RX(element, wires=idx)
        


# In[6]:


def RZ_layer(w):
    for idx, element in enumerate(w):
        qml.RZ(element, wires=idx)


# In[7]:



def U1_layer(w):
    for idx, element in enumerate(w):
        qml.U1(element, wires=idx)


# In[8]:


def U2_layer(w):
    for idx, element in enumerate(w):
        qml.U2(element,element, wires=idx)       


# In[9]:



def U3_layer(w):
   for idx, element in enumerate(w):
       qml.U3(element,element,element, wires=idx)


# In[10]:


def PhaseShift_layer(w):
    for idx, element in enumerate(w):
        qml.PhaseShift(element, wires=idx)    


# In[11]:


def entangling_layer_CNOT(nqubits,flag=True):
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])

    if flag == True:
        qml.CNOT(wires=[n_qubits-1,0])
    else:
        qml.CNOT(wires=[0,n_qubits-1])
        


# In[12]:


def entangling_layer_CZ(nqubits,flag=True):
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CZ(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CZ(wires=[i, i + 1])

    if flag == True:
        qml.CZ(wires=[n_qubits-1,0])
    else:
        qml.CZ(wires=[0,n_qubits-1])


# In[13]:



def entangling_layer_CRX(nqubits,flag=True):
   
    # print("entangling")
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CRX(0.1,wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CRX(0.1,wires=[i, i + 1])

    if flag == True:
        qml.CRX(0.1,wires=[n_qubits-1,0])
    else:
        qml.CRX(0.1,wires=[0,n_qubits-1])


# In[ ]:




