#!usr/bin/python
# -*- coding:UTF-8 -*-
#Created on: 20191210
#author: Qinghong Chen
 
#-------------------------FUNCTION---------------------------#
import sklearn.metrics as metrics
import numpy as np
from collections import defaultdict
np.random.seed(304)

def SGD(train,test,N,M,eta,K,lambda_1,lambda_2,Step):
    # train: train data
    # test: test data
    # N:the number of user
    # M:the number of item
    # eta: the learning rata
    # K: the number of latent factor
    # lambda_1,lambda_2: regularization parameters
    # Step: the max iteration
    U = np.random.normal(0, 0.1, (N, K))
    V = np.random.normal(0, 0.1, (M, K))
    rmse = []
    loss = []
    for ste in range(Step):
        los = 0.0
        for data in train:
            u = data[0]
            i = data[1]
            r = data[2]
            e = r-np.dot(U[u],V[i].T)
            U[u] = U[u]+eta*(e*V[i]-lambda_1*U[u])
            V[i] = V[i]+eta*(e*U[u]-lambda_2*V[i])
            los = los+0.5*(e**2+lambda_1*np.square(U[u]).sum()+lambda_2*np.square(V[i]).sum())
        loss.append(los)
        rms=RMSE(U,V,test)
        rmse.append(rms)
        print("In step {ste}, rmse is {rmse}, loss is {loss}".format(ste=ste, rmse=rms, loss=los))
    return U,V

def Load_data(filedir,ratio):
    user_set={}
    item_set={}
    N=0 #the number of user
    M=0 #the number of item
    u_idx=0
    i_idx=0
    data=[]
    f = open(filedir)
    for line in f.readlines():
        u,i,r,t=line.split()
        if int(u) not in user_set:
            user_set[int(u)]=u_idx
            u_idx+=1
        if int(i) not in item_set:
            item_set[int(i)]=i_idx
            i_idx+=1
        data.append([user_set[int(u)],item_set[int(i)],int(r)])
    f.close()
    N=u_idx
    M=i_idx
    np.random.shuffle(data)
    train=data[0:int(len(data)*ratio)]
    test=data[int(len(data)*ratio):]
    return N,M,train,test

def RMSE(U,V,test):
    count=len(test)
    sum_rmse=0.0
    for t in test:
        u=t[0]
        i=t[1]
        r=t[2]
        pr=np.dot(U[u],V[i].T)
        sum_rmse+=np.square(r-pr)
    rmse=np.sqrt(sum_rmse/count)
    return rmse

def AUC(user_inter_test,user_item_pred):
    users = user_inter_test.keys()
    auc = []
    for user in users:
        # print(user_inter_test[user],user_item_pred[user])
        pred = list(user_item_pred[user].values())
        label = []
        for item in list(user_item_pred[user].keys()):
            if item in user_inter_test[user]:
                label.append(1)
            else:
                label.append(0)
        try:
            au = metrics.roc_auc_score(y_true = label, y_score = pred)
        except Exception:
            au = 0
        auc.append(au)
    return sum(auc)/len(auc)

def Get_user_item_pred(U,V,test,user_inter_test):
    user_item_pred = defaultdict(dict)
    for t in test:
        u = t[0]
        i = t[1]
        pr = np.dot(U[u], V[i].T)
        user_item_pred[u][i] = pr
    return user_item_pred

def Get_user_interaction(train,test):
    data_set = [train, test]
    user_inter_test = defaultdict(list)
    user_inter_train = defaultdict(list)
    user_inter = [user_inter_train, user_inter_test]
    for idx in range(2):
        for record in data_set[idx]:
            u = record[0]
            i = record[1]
            r = record[2]
            if r >=4:
                user_inter[idx][u].append(i)
    return user_inter_train, user_inter_test

def main():
    dir_data = "./u.data"
    ratio = 0.8
    N,M,train,test = Load_data(dir_data,ratio)
    user_inter_train, user_inter_test = Get_user_interaction(train, test)

    eta = 0.005
    K = 8
    lambda_1 = 0.1
    lambda_2 = 0.1
    Step = 20
    U,V = SGD(train,test,N,M,eta,K,lambda_1,lambda_2,Step)

    user_item_pred = Get_user_item_pred(U, V, test, user_inter_test)
    auc = AUC(user_inter_test, user_item_pred)
    print('auc: {}'.format(auc))
         
if __name__ == '__main__': 
    main()
