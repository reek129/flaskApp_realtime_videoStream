# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 18:54:41 2021

@author: reekm
"""

import pennylane as qml


def feature_map1(x,n_qubits,feature_depth):
    for _ in range(feature_depth):
        for i in range(n_qubits):
            qml.RY(x[i],wires= i)
            qml.RZ(x[i], wires=i)
        for i in range(n_qubits - 1, 0, -1):
            qml.CNOT(wires=[i, i-1])
        qml.RY(x[1], wires=1) 
        qml.RZ(x[1], wires=1)
        
def feature_map2(x,n_qubits,feature_depth):
    for _ in range(feature_depth):
        for i in range(n_qubits):
            qml.RX(x[i],wires= i)
            qml.RZ(x[i], wires=i)
        
        for control in range(n_qubits-1, 0, -1):
            target = control - 1
            qml.RX(x[target],wires= target)
            qml.CNOT(wires=[control, target])
            qml.RX(x[target],wires= target)
            
        for i in range(n_qubits):
            qml.RX(x[i],wires= i)
            qml.RZ(x[i],wires= i)
            

def feature_map3(x,n_qubits,feature_depth): 
    
    for _ in range(feature_depth):
        for i in range(n_qubits):
            qml.RX(x[i], wires=i)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])
                qml.U1(x[i] * x[j],wires= j)
                qml.CNOT(wires=[i, j])
        

def feature_map4(x,num_qubits,reps): 
    
    for _ in range(reps):
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        for i in range(num_qubits - 1):
            qml.CZ(wires=[i, i+1])
        for i in range(num_qubits):
            qml.RX(x[i],wires= i)
        for i in range(num_qubits-1, 0, -1):
            qml.CZ(wires=[i, i-1])
        for i in range(num_qubits):
            qml.Hadamard(wires=i)


def feature_map5(x,num_qubits,reps): 
    
    for _ in range(reps):
        for i in range(num_qubits):
            qml.RX(x[i],wires= i)
            qml.RZ(x[i],wires= i)
        for i in range(num_qubits-1, 0, -1):
            qml.CNOT(wires=[i, i-1])

def feature_map6(x,num_qubits,reps): 
    
    for _ in range(reps):
        for i in range(num_qubits):
            qml.RX(x[i], wires=i)
            qml.RZ(x[i], wires=i)
        for i in range(num_qubits-1, 0, -1):
            qml.CZ(wires=[i, i-1])

def feature_map7(x,num_qubits,reps): 
    
    for _ in range(reps):
        for i in range(num_qubits):
            qml.RX(x[i], wires=i)
            qml.RZ(x[i], wires=i)
        for i in range(num_qubits-1, 0, -1):
            qml.RZ(x[i-1], wires=i-1)
            qml.CNOT(wires=[i, i-1])
            qml.RZ(x[i-1],wires= i-1)

def feature_map8(x,num_qubits,reps): 
    
    for _ in range(reps):
        for i in range(num_qubits):
            qml.RX(x[i],wires= i)
            qml.RZ(x[i],wires= i)
        for i in range(num_qubits-1, 0, -1):
            qml.RX(x[i-1],wires= i-1)
            qml.CNOT(wires=[i, i-1])
            qml.RX(x[i-1],wires= i-1)

def feature_map9(x,num_qubits,reps): 
    
    for _ in range(reps):
        for i in range(num_qubits):
            qml.RX(x[i],wires= i)
            qml.RZ(x[i],wires= i)
        for control in range(num_qubits-1, -1, -1):
            for target in range(num_qubits-1, -1, -1):
                if control != target:
                    qml.RZ(x[target],wires= target)
                    qml.CNOT(wires=[control, target])
                    qml.RZ(x[target], wires=target)
    

def feature_map10(x,num_qubits,reps): 
    
    
    for _ in range(reps):
        for i in range(num_qubits):
            qml.RX(x[i], wires=i)
            qml.RZ(x[i], wires=i)
        for control in range(num_qubits-1, -1, -1):
            for target in range(num_qubits-1, -1, -1):
                if control != target:
                    qml.RX(x[target],wires= target)
                    qml.CNOT(wires=[control, target])
                    qml.RX(x[target], wires=target)

def feature_map11(x,num_qubits,reps): 
    
    for _ in range(reps):
        for i in range(num_qubits):
            qml.RX(x[i], wires=i)
            qml.RZ(x[i], wires=i)
        for control in range(num_qubits-1, -1, -1):
            for target in range(num_qubits-1, -1, -1):
                if control != target:
                    qml.RX(x[target], wires=target)
                    qml.CNOT(wires=[control, target])
                    qml.RX(x[target], wires=target)

def feature_map12(x,num_qubits,reps): 
    
    for _ in range(reps):
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        for i in range(num_qubits - 1):
            qml.CZ(wires=[i, i+1])
        for i in range(num_qubits):
            qml.RX(x[i], wires=i)

def feature_map13(x,num_qubits,reps): 
    
    for _ in range(reps):
        for i in range(num_qubits):
            qml.RY(x[i],wires= i)
        for i in range(num_qubits - 1, 0, -1):
            qml.CZ(wires=[i, i-1])
        qml.CZ(wires=[num_qubits-1, 0])
        for i in range(num_qubits):
            qml.RY(x[i], wires=i)

def feature_map14(x,num_qubits,reps): 
    
    for _ in range(reps):
        for i in range(num_qubits):
            qml.RY(x[i],wires= i)
            qml.RZ(x[i], wires=i)
        for i in range(num_qubits - 1, 0, -1):
            qml.CNOT(wires=[i, i-1])
           
        
def feature_map15(x,num_qubits,reps): 
    
    for _ in range(reps):
        for i in range(num_qubits):
            qml.RY(x[i], wires=i)
            qml.RZ(x[i], wires=i)
        for i in range(num_qubits - 1, 0, -1):
            qml.CZ(wires=[i, i-1])
        qml.RY(x[1],wires= 1)
        qml.RZ(x[1], wires=1)

def feature_map16(x,num_qubits,reps): 
    
    for _ in range(reps):
        for i in range(num_qubits):
            qml.RX(x[i], wires=i)
            qml.RZ(x[i], wires=i)
        qml.RX(x[0], wires=0)
        qml.CNOT(wires=[num_qubits - 1, 0])
        qml.RX(x[0], wires=0)
        for i in range(num_qubits-2, -1, -1):
            qml.RX(x[i+1],wires= i+1)
            qml.CNOT(wires=[i, i+1])
            qml.RX(x[i+1],wires= i+1)


def feature_map17(x,num_qubits,reps): 
    
    for _ in range(reps):
        for i in range(num_qubits):
            qml.RX(x[i], wires=i)
            qml.RZ(x[i], wires=i)
        qml.RZ(x[0], wires=0)
        qml.CNOT(wires=[num_qubits - 1, 0])
        qml.RZ(x[0], wires=0)
        for i in range(num_qubits-2, -1, -1):
            qml.RZ(x[i+1],wires= i+1)
            qml.CNOT(wires=[i, i+1])
            qml.RZ(x[i+1],wires= i+1)



def feature_map18(x,num_qubits,reps): 
    
    for _ in range(reps):
        for i in range(num_qubits):
            qml.RX(x[i], wires=i)
            qml.RZ(x[i], wires=i)
        for control in range(num_qubits - 1, 0, -1):
            target = control - 1
            qml.RX(x[target], wires=target)
            qml.CNOT(wires=[control, target])
            qml.RX(x[target],wires= target)



def feature_map19(x,num_qubits,reps): 
    
    for _ in range(reps):
        for i in range(num_qubits):
            qml.RY(x[i], wires=i)
        qml.CNOT(wires=[num_qubits-1, 0])
        for i in range(num_qubits-1):
            qml.CNOT(wires=[i, i+1])
        for i in range(num_qubits):
            qml.RY(x[i], wires=i)
        qml.CNOT(wires=[num_qubits - 1, num_qubits - 2])
        qml.CNOT(wires=[0, num_qubits - 1])
        for i in range(1, num_qubits - 1):
            qml.CNOT(wires=[i, i-1])
            
def feature_map20(x,num_qubits,reps): 
    
    for _ in range(reps):               # Exp 20
        for i in range(num_qubits):
            qml.RZ(x[i],wires= i)
            qml.RX(x[i], wires=i)
        for control in range(num_qubits-1, 0, -1):
            target = control - 1
            qml.RX(x[target],wires= target)
            qml.CNOT(wires=[control, target])
            qml.RX(x[target], wires=target)
        for i in range(num_qubits):
            qml.RZ(x[i],wires= i)
            qml.RX(x[i],wires= i)