import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights
import pennylane as qml
from pennylane import numpy as np

from layers import RY_layer,U1_layer,RZ_layer,RX_layer,U2_layer,U3_layer,PhaseShift_layer,H_layer
from layers import entangling_layer_CNOT,entangling_layer_CZ,entangling_layer_CRX

from featureMap import feature_map1,feature_map2,feature_map3,feature_map4,feature_map5
from featureMap import feature_map6,feature_map7,feature_map8,feature_map9,feature_map10
from featureMap import feature_map11,feature_map12,feature_map13,feature_map14,feature_map15
from featureMap import feature_map16,feature_map17,feature_map18,feature_map19,feature_map20



config_dict1 ={
    'Model 28_v7':  [1,'70',1,'3',1,'02',0,3,3,'Model_28_v7_wd',100]
}
n_qubits = 4
#
n_classes=2

dev = qml.device("default.qubit", wires=n_qubits)

class Quantum_circuit():
    
    ip = 1 
    front_layer = '70'
    op = 0
    last_layer = '3'
    entanglement_layer = 0
    middle_layer= '02'
    measurement = 2
    fmap_depth = 3
    var_depth = 3
    model_id = 'default'
    featureMap_id = 100
    n_qubits =4
    
    def __init__(self,ip,front_layer,op,last_layer,entanglement_layer,middle_layer,measurement,fmap_depth,var_depth,model_id,featureMap_id):
        self.ip = ip
        self.front_layer = front_layer
        self.op = op
        self.last_layer = last_layer
        self.entanglement_layer = entanglement_layer
        self.middle_layer = middle_layer
        self.measurement = measurement
        self.fmap_depth = fmap_depth
        self.var_depth = var_depth
        self.model_id = model_id 
        self.featureMap_id = featureMap_id
        self.n_qubits = n_qubits
        
        
    def get_layer(self,layerId,weights):
        if layerId == '0':
            RY_layer(weights)
        elif layerId =='1':
            U1_layer(weights)
        elif layerId =='2':
            RZ_layer(weights)
        elif layerId =='3':
            RX_layer(weights)
        elif layerId =='4':
            U2_layer(weights)
        elif layerId =='5':
            U3_layer(weights)
        elif layerId =='6':
            PhaseShift_layer(weights)
        elif layerId =='7':
            H_layer(n_qubits)
            
    def get_entanglement_layer(self,id,flag=False):
        if id == 0:
            entangling_layer_CNOT(self.n_qubits,flag)
        if id ==1:
            entangling_layer_CZ(self.n_qubits,flag)
        if id == 2:
            entangling_layer_CRX(self.n_qubits,flag)
            
    def front_layers(self,w):
        for x in self.front_layer:
            self.get_layer(x,w)
            
    def last_layers(self,w):
        for x in self.last_layer:
            self.get_layer(x,w)
            
    def get_var_layer(self,weights):
        self.get_entanglement_layer(self.entanglement_layer)
        for x in self.middle_layer:
            self.get_layer(x,weights)
            
            
    def get_var_layer2(self,k,weights):
        self.get_entanglement_layer(self.entanglement_layer)
        for i,j in enumerate(self.middle_layer):
            self.get_layer(j,weights[(3*i)+k])
            
    def get_expectation_value(self,measurementId):
        gate_set = [qml.PauliX, qml.PauliY, qml.PauliZ]
        exp_val = [qml.expval(gate_set[measurementId](position)) for position in range(self.n_qubits)]
        return exp_val
    
    

    def get_feature_map(self,x,qubits,reps):
        featureMap_id = self.featureMap_id
        if featureMap_id == 1:
            feature_map1(x,qubits,reps)
        elif featureMap_id == 2:
            feature_map2(x,qubits,reps)
        elif featureMap_id == 3:
            feature_map3(x,qubits,reps)
        elif featureMap_id == 4:
            feature_map4(x,qubits,reps)
        elif featureMap_id == 5:
            feature_map5(x,qubits,reps)
        elif featureMap_id == 6:
            feature_map6(x,qubits,reps)
        elif featureMap_id == 7:
            feature_map7(x,qubits,reps)
        elif featureMap_id == 8:
            feature_map8(x,qubits,reps)
        elif featureMap_id == 9:
            feature_map9(x,qubits,reps)
        elif featureMap_id == 10:
            feature_map10(x,qubits,reps)
        elif featureMap_id == 11:
            feature_map11(x,qubits,reps)
        elif featureMap_id == 12:
            feature_map12(x,qubits,reps)
        elif featureMap_id == 13:
            feature_map13(x,qubits,reps)
        elif featureMap_id == 14:
            feature_map14(x,qubits,reps)
        elif featureMap_id == 15:
            feature_map15(x,qubits,reps)
        elif featureMap_id == 16:
            feature_map16(x,qubits,reps)
        elif featureMap_id == 17:
            feature_map17(x,qubits,reps)
        elif featureMap_id == 18:
            feature_map18(x,qubits,reps)
        elif featureMap_id == 19:
            feature_map19(x,qubits,reps)
        elif featureMap_id == 20:
            feature_map20(x,qubits,reps)

@qml.qnode(dev, interface="torch")
def quantum_net(inputs, weights):
#    print(config_dict1[key2])
#    print(len(config_dict1[key2]))
    qc = Quantum_circuit(ip,front_layer,op,last_layer,entanglement_layer,middle_layer,measurement,fmap_depth,var_depth,model_id,featureMap_id )
    
    if qc.ip ==1:
        qc.front_layers(inputs)
    
    qc.get_feature_map(inputs,n_qubits,int(qc.fmap_depth))
        
    for k in range(int(qc.var_depth)):
        qc.get_var_layer(weights[k])
        
    for k in range(int(qc.var_depth)):
        qc.get_var_layer2(k,weights)
        
    qc.get_entanglement_layer(qc.entanglement_layer,True)
    
    if qc.op == 1:
        qc.last_layers(inputs)
        
    exp_vals = qc.get_expectation_value(int(qc.measurement))
    return tuple(exp_vals)


class DressedQuantumNet2(nn.Module):
    

    def __init__(self,num_ftrs,key):
        

        super().__init__()
        print(num_ftrs)
        self.pre_net = nn.Linear(num_ftrs, n_qubits)
#        weight_shapes = {"weights": (3, n_qubits, 3)}
        weight_shapes={'weights':(var_depth * len(middle_layer) ,n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(quantum_net,weight_shapes)
        self.post_net = nn.Linear(n_qubits, n_classes)
        self.key = key
        print(self.key)
#        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))
       

    def forward(self, input_features):
        
        pre_out = self.pre_net(input_features)
        q_out = self.qlayer(pre_out)
        
#        q_in = torch.tanh(pre_out) * np.pi / 2.0
#
#        q_out = torch.Tensor(0, n_qubits)
#        
#        q_out = q_out.to(device)
#        
#        for elem in q_in:
#            q_out_elem = quantum_net2(elem, self.q_params).float().unsqueeze(0)
#            
#            q_out = torch.cat((q_out, q_out_elem))

        return self.post_net(q_out)

def quantum_model2():
    global ip,front_layer,op,last_layer,entanglement_layer,middle_layer,measurement,fmap_depth,var_depth,model_id,featureMap_id 
    # print("config_dict")
    for i in config_dict1.keys():
        key = i

    # print(key)
    ip,front_layer,op,last_layer,entanglement_layer,middle_layer,measurement,fmap_depth,var_depth,model_id,featureMap_id = config_dict1[key]

    model_hybrid = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model_hybrid.parameters():
            param.requires_grad = False
            
    num_ftrs = model_hybrid.fc.in_features
    model_hybrid.fc = DressedQuantumNet2(num_ftrs,key)
    
        
    return model_hybrid


