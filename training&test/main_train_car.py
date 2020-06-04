import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data

from prune import *
from heapq import nsmallest
from operator import itemgetter
import numpy as np
from DfpNet_design_relu import TurbNetG 
# from UNet import TurbNetG
import sys

device = torch.device('cuda')

file_name = 'result_test' + str(sys.argv[1]) +'.txt'

best_loss_val = np.infty
best_batch = 0

class MyDataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)



def adjust_learning_rate(optimizer, decay_rate=.9):
    """Sets the learning rate to the initial LR decayed by 5 every 100 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()
        
    def reset(self):
        self.filter_ranks = {}
        
    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}
        
        activation_index = 0
        temp_a = [] 
        ptr = 0
        m_length = []
        total_length = len(self.model._modules.items())
        for layer, (name, module) in enumerate(self.model._modules.items()):
            m_length.append(len(module))
            if layer > 7:
                x = torch.cat([x, temp_a[7-layer]], 1)
            for slayer, (sname, smodule)  in enumerate(module._modules.items()):
                # print (slayer)
                x = smodule(x)
                if isinstance(smodule, torch.nn.modules.conv.Conv2d):
                    # print ("yes")
                    x.register_hook(self.compute_rank)
                    self.activations.append(x)
                    self.activation_to_layer[activation_index] = slayer + ptr
                    activation_index += 1
            ptr = ptr + len(module)
            if layer < 6:
                temp_a.append(x)

            # print ("yes")
  
        return x        
        # return self.model.classifier(x.view(x.size(0), -1))
    
    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        
        values = torch.sum((activation * grad), dim=0).\
            sum(dim=1).sum(dim=1).data
        values /= (activation.size(0) * activation.size(2) * activation.size(3))
        
        if activation_index not in self.filter_ranks:
            # print ("yes2")
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_().to(device)
        
        self.filter_ranks[activation_index] += values
        self.grad_index += 1
        
    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
                
        return nsmallest(num, data, itemgetter(2))
    
    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i]).cpu()
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v
        
    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        # print (filters_to_prune)
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)
        
        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i 

        filters_to_prune = []
        for l in filters_to_prune_per_layer:

            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))
        
        return filters_to_prune
   

from input2_new import flow_train_inputs
from input2_new import flow_test_inputs
from LossFunc import MyLoss


class PrunningFineTuner_CNN:
    def __init__(self, model , batchsize):
        boun_train,sflow_train = flow_train_inputs()
        boun_test,sflow_test = flow_test_inputs()
        #boun_train,sflow_train = torch.randn(16,1,128,256),torch.randn(16,2,128,256)
        #boun_test,sflow_test = torch.randn(16,1,128,256),torch.randn(16,2,128,256)


        train_dataset = MyDataset(boun_train, sflow_train)
        test_dataset = MyDataset(boun_test, sflow_test)
        
        self.train_data_loader = data.DataLoader(
            dataset=train_dataset,      # torch TensorDataset format
            batch_size=batchsize,      # mini batch size
            shuffle=True,               
            num_workers=0,             
        )

        self.test_data_loader = data.DataLoader(
            dataset=test_dataset,      # torch TensorDataset format
            batch_size=1,      # mini batch size #total 10
            shuffle=True,               
            num_workers=0,             
        )



        self.loss_val = 0.0
        self.model = model
        self.criterion = MyLoss() #torch.nn.L1Loss() #MyLoss()
        self.prunner = FilterPrunner(self.model)
        self.model.train()
        
    def test(self,batch_num):
        self.model.eval()
        loss_test = 0.0
        
        for _, (batch, label) in enumerate(self.test_data_loader):
            batch = batch.to(device)
            label = label.to(device)
            output = self.model(batch)

            loss_ = self.criterion(output, label)

            loss_test += loss_.item()
        

        loss_test = loss_test / len(self.test_data_loader)
        global best_loss_val,best_batch
        if loss_test < best_loss_val:
            torch.save(self.model, "model_val_best")
            best_loss_val = loss_test
            best_batch = batch_num 
        print("loss_val : ", loss_test)
        with open(file_name,'a') as file_obj:
            file_obj.write(" loss_val : "+ str(loss_test) + '\n')
        self.model.train()
    
    def train(self, optimizer = None, epoches = 10):
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        
        for i in range(epoches):
            if (i+1) % 25 == 0:
                adjust_learning_rate(optimizer)    
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.loss_val = self.loss_val / len(self.train_data_loader)
            print("Loss_train : ", self.loss_val)
            with open(file_name,'a') as file_obj:
                file_obj.write("Epoch: " + str(i) + '\n')
                file_obj.write("Loss_train : " + str(self.loss_val) + '\n')
            self.test(i)
            if i%10 == 0:
                torch.save(self.model, "model_test"+ str(sys.argv[1]))   
# self.test()
        print("Finished fine tuning.")
    
    def train_epoch(self, optimizer = None, rank_filters = False):
        for _, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch.to(device), label.to(device), rank_filters)
    
    def train_batch(self, optimizer, batch, label, rank_filters):
        self.model.zero_grad()
        if rank_filters:
            output = self.prunner.forward(batch)
            loss = self.criterion(output, label)
            self.loss_val += loss.item()
            #optimizer.zero_grad()
            loss.backward()
        else:
            loss = self.criterion(self.model(batch), label)
            self.loss_val += loss.item()
            optimizer.zero_grad()           
            loss.backward()

            optimizer.step()
    
    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters = True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)
    
    def total_num_filters(self):
        filters = 0
        for name, module in self.model._modules.items():
            for layer in module:
                # print (layer)
                if isinstance(layer, torch.nn.modules.conv.Conv2d):
                    filters = filters + layer.out_channels
        # print (filters)
        return filters
    
    def prune(self):
        # self.test()
        self.model.train()
        
        for param in self.model.parameters():
            param.requires_grad = True
            
        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 8
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
        
        iterations = int(iterations * 9.0 / 10)
        print("Number of prunning iterations to reduce 90% filters", iterations)
        
        Layers_Prunned = []
        # Acc = []
        for epoch_no in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            print (prune_targets)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1
            
            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                
                model = prune_conv_layer(model, layer_index, filter_index)
            
            print (model)
            self.model = model.to(device)
            
            message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            # self.test()
            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.train(optimizer, epoches = 40)
            torch.save(model, "./model_prunned/model_prunned_"+str(epoch_no))
            
            # self.model.eval()
            # correct = 0
            # total = 0
            # for i, (batch, label) in enumerate(self.test_data_loader):
            #     batch = batch.to(device)
            #     output = self.model(Variable(batch))
            #     pred = output.data.max(1)[1]
            #     correct += pred.cpu().eq(label).sum()
            #     total += label.size(0)
            # acc = float(correct) / total
            # Acc.append(acc)
            # Layers_Prunned.append(layers_prunned)
            
        print("Finished. Going to fine tune the model a bit more")
        self.train(optimizer, epoches = 15)
        torch.save(model, "model_final_prunned")
        # return Acc  Layers_Prunned
        return Layers_Prunned



#model = TurbNetG(channelExponent=5, dropout=0.0).to(device)
model = TurbNetG(channelExponent=5).to(device)
print(model)
bz = 16
fine_tuner = PrunningFineTuner_CNN(model,bz)


opt = optim.Adam(model.parameters(), lr=4e-4)
fine_tuner.train(optimizer = opt, epoches = 4)

torch.save(model, "model_test"+ str(sys.argv[1]))
"""


model = torch.load("model_CBAM").to(device)
fine_tuner = PrunningFineTuner_CNN(model, 8)
# Acc, 
Layers_Prunned = fine_tuner.prune()
"""




