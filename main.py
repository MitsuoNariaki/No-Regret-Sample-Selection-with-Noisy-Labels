import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
#import torchvision.models as models
import tensorflow as tf
import numpy as np
import random
import dataset
import model



def train(epoch):
    Model.train()
    #scheduler.step()
 
    steps = len(ds_selected)//batch_size
    for step, (images, labels) in enumerate(dataloader_selelected, 1):
 
        images, labels = images.to(device), labels.to(device)
 
        optimizer.zero_grad()
        outputs = Model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
 
        if step % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch, epochs, step, steps, loss.item()))
            
def softmax(Llist):
    exp_x = np.exp(Llist)    
    y = exp_x / np.array([np.sum(exp_x,axis=1)]).T    
    return np.max(y,axis=1)


def cal_loss(epoch):
    Model.eval()
    correct = 0
    total = 0
    label_RightorWrong = []
    with torch.no_grad():
        p_list = []
        for step,(images, labels) in enumerate(dataloader_prob,1):
            
            labelinf = np.zeros(len(labels))
            images, labels = images.to(device), labels.to(device)
            outputs = Model(images)
           
            _, predicted = torch.max(outputs.data, 1)
            
            p = softmax([list(map(float,i)) for i in list(outputs)])         
            p_list.extend(p)
            
            prelist = list(map(int,predicted))
            lablist = list(map(int,labels))
            
            labelinf[np.where(np.array(prelist) == np.array(lablist))[0]] = 1
            labelinf[np.where(np.array(prelist) != np.array(lablist))[0]] = -1
            
            label_RightorWrong.extend(labelinf)

        train_acc = len(np.where(np.array(label_RightorWrong)==1)[0])/len(label_RightorWrong)
        loss_list = (1 - np.array(label_RightorWrong)*(1 - np.array(p_list)))/2
        
        print("Tra Acc : %.4f" % (train_acc))
    
    return np.array(loss_list),train_acc

def eval(epoch):
    Model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in dataloader_val:
            images, labels = images.to(device), labels.to(device)
 
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
 
    print("Val Acc : %.4f" % (correct/total))
    
    return correct/total

def test(epoch):
    Model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in dataloader_test:
            images, labels = images.to(device), labels.to(device)
 
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
 
    print("Test Acc : %.4f" % (correct/total))
    
    return correct/total


def WAA(x,loss_list):
    w_Mom = sum(w*pow(beta,loss_list))
    for i in range(len(x)):
        w_Child = w[i]*pow(beta,loss_list[i])
        w[i] = w_Child/w_Mom
    
    return w

def projection(x,k):
    ranking = np.argsort(x)[::-1]
    x_sort = np.sort(x)[::-1]
    
    if max(x_sort) < 1/k:
        y = x_sort
    else:
        y = x_sort
        for i in range(n):
            if y[i] > 1/k:
                y[i] = 1/k
                y[i+1:] = y[i+1:]*((1-(i+1)/k)/sum(y[i+1:]))
            else:
                pass
    
    y_sort_reset = np.zeros(n)

    for j in range(n):
        y_sort_reset[ranking[j]] = y[j]
    
    return y_sort_reset

def conv(conv_comb,k):
    
    conv_comb_len = len(conv_comb)
    zero_point = np.where(conv_comb==0)[0]

    k_point = np.where(abs(conv_comb-np.linalg.norm(conv_comb,ord=1)*(1/k))<=1e-17)[0]
    
    endpoint = np.zeros(conv_comb_len) - 1
    endpoint[zero_point] = 0
    endpoint[k_point] = 1/k
    
    undecided_list = np.where(endpoint==-1)[0]
    k_undecided_len = k - len(k_point)

    other_k_point = undecided_list[:k_undecided_len]
 
    endpoint[other_k_point] = 1/k
    endpoint = np.where(endpoint==-1,0,endpoint)
 
    return endpoint

def decomposition(conv_comb,k,n):

    c_list = []
    a_list = []
    l = 0
    a = 0
    while max(abs(conv_comb)) >= 1e-17:
        
        if l > n or a < 0:
            print("error")
            break
        if l % 10000==0:
           
            print("step:"+str(l))
        else:
            pass
        
        c = conv(conv_comb,k)
        m = min(conv_comb[np.where(c==1/k)[0]])
        M = max(conv_comb[np.where(c==0)[0]])
        a = min(k*m,np.linalg.norm(conv_comb,ord=1)-k*M)
        
        conv_comb = conv_comb - a*c
        conv_comb[conv_comb<=1e-17] = 0
        
        c_list.append(c)
        a_list.append(a)
        l = l + 1

    c_number = np.random.choice(range(len(c_list)),p=a_list/sum(np.array(a_list)))

    c_selected = k*c_list[c_number]
    
    
    return c_selected


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

num_classes = 10
batch_size = 128
epochs = 100
val_ratio = 0.1
noise_ratio = 0.2
kset = 0.8

dataloader_prob, dataloader_val, dataloader_test, x_train, y_train_rand = dataset.noisy_label_dataset(val_ratio,noise_ratio)
n = len(x_train)
k = int(kset*n)
eta = np.sqrt(8*np.log(n)/epochs)
beta = np.exp(-eta)

#Model = model.Cifar10Model().to(device)
Model = model.ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(Model.parameters(), lr=0.1)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

train_acclist = []
val_acclist = []
test_acclist = []



#===algorithm setting===

#============================================
algorithm = "WAA"
w = np.array([1/n for i in range(n)])
#============================================

#============================================
#algorithm = "FPL"
#cumulative_loss = np.zeros(n)
#xt = np.array(random.sample(range(0,n),k=k))
#============================================




for epoch in range(1, epochs+1):
    print("\n--- Epoch : %2d ---" % epoch)
    print("lr : %f" % optimizer.param_groups[0]['lr'])    
    
    if algorithm == "WAA":
        y = projection(w,k)
        c = decomposition(y,k,n)
        ct = np.where(c==1)[0]

        ds_selected = data.TensorDataset(torch.from_numpy(x_train[ct]), torch.from_numpy(y_train_rand[ct]))
        dataloader_selelected = data.DataLoader(dataset=ds_selected, batch_size=batch_size, shuffle=True)
        train(epoch)
        loss_list,train_acc = cal_loss(epoch)

        w = WAA(y,loss_list)
        
    
    elif algorithm == "FPL":
        xt = np.sort(xt)
        c = np.zeros(n)
        c[xt] = 1
        
        ds_selected = data.TensorDataset(torch.from_numpy(x_train[xt]), torch.from_numpy(y_train_rand[xt]))
        dataloader_selelected = data.DataLoader(dataset=ds_selected, batch_size=batch_size, shuffle=True)
        train(epoch)
        loss_list,train_acc = cal_loss(epoch)
        cumulative_loss = cumulative_loss + loss_list
        perturbation = np.random.normal(0,0.00001,(n))
        virtual_loss = cumulative_loss + eta*perturbation 
        xt = np.argsort(virtual_loss)[:k]
    
    val_acc = eval(epoch)
    test_acc = test(epoch)
    
    train_acclist.append(train_acc)
    val_acclist.append(val_acc)
    test_acclist.append(test_acc)
    
    
