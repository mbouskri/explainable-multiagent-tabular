import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np
from torchmetrics.functional import accuracy
import time
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from multiprocessing import Process, Barrier, Value, Manager
from ctypes import c_char_p
from sklearn.utils.class_weight import compute_class_weight


from models import DiscNN, ClassifierNN, AttentionNN
from datasets_L import openmldataset_tt_multi,openmldataset_val_multi
from loss_function import Cos_loss
from utils import get_dataset_id_multi, get_dataset_multi, split_tensors,get_dataset_multi_from_file

torch.set_default_dtype(torch.float32)

class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ORANGE = '\033[38;5;209m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



def train_(scheduler_D,scheduler_C, scheduler_A,model_d, model_c, model_a, traindata, optimizer_D,optimizer_C, optimizer_A, criterion,bce_f, batch_size, device):
    """
    Function performing one epoch training loop
    Returns all metrics and execution time
    """
    a=time.time()
    model_d.to(device)
    model_d.train()
    model_c.to(device)
    model_c.train()
    model_a.to(device)
    model_a.train()
    train_dataloader = DataLoader(traindata, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    total_roc = 0.0
    num_batches = 0

    for batch_idx, (x1, x2, l, y_b, y) in enumerate(train_dataloader):
        if batch_idx == 0:
            batch_loss = 0.0
            batch_acc = 0.0
            batch_f1 = 0.0
            batch_roc = 0.0

        for k in range(x2.size()[1]):
            # for k in range(n), n=1,5,10
            if k == 0:
                running_loss = 0.0
                running_acc = 0.0
                running_f1 = 0.0
                running_roc = 0.0

            # Discriminator forward pass on x1
            z1 = model_d(x1.to(device)) # Add squeeze if not using rnn

            # Discriminator forward pass on x2k
            z2=[]
            Z2=model_d(x2.to(device)[:,k,:])

            # Attention forward pass on z1
            attended1, weights1 = model_a(z1)
            # Attention forward pass on Z2
            attended2, weights2 = model_a(Z2)

            #Residual connection
            za1=torch.add(z1,attended1)
            Za2=torch.add(Z2,attended2)

            za1=nn.BatchNorm1d(za1.size(-2))(za1)
            Z2=nn.BatchNorm1d(Za2.size(-2))(Za2)#change Za2 to Z2 if not attending

            # Classifier forward pass
            y_ = model_c(za1)


            if torch.any(torch.isnan(y_)): print("y_ has NaN") # Checking for NaNs

            # Distance Loss
            cos_loss = criterion(za1,Z2,l[:,k])
            #Binary Cross Entropy Loss
            bce = bce_f(y_,y_b.reshape(y_b.size(0),1).to(device))
            # Combining losses
            j_loss = cos_loss + bce


            with torch.no_grad():

                train_acc=accuracy(y_.to(device), y_b.reshape(y_b.size(0),1).to(device),'binary')
                running_acc += train_acc.item()

                f1 = f1_score(y_.to(device), y_b.reshape(y_b.size(0),1).to(device), task="binary")
                running_f1 += f1.item()

                try:
                    roc = roc_auc_score(y_b.reshape(y_b.size(0),1).cpu(),y_.cpu())
                    running_roc += roc.item()
                except ValueError:
                    running_roc += 0.5

            running_loss += j_loss.item()
            optimizer_D.zero_grad()
            optimizer_A.zero_grad()
            optimizer_C.zero_grad()

            j_loss.backward()
            optimizer_D.step()
            optimizer_A.step()
            optimizer_C.step()

        batch_loss += running_loss / x2.size()[1]
        batch_acc += running_acc / x2.size()[1]
        batch_f1 += running_f1 / x2.size()[1]
        batch_roc += running_roc / x2.size()[1]
        num_batches += 1

    total_loss = batch_loss
    total_acc = batch_acc
    total_f1 = batch_f1
    total_roc = batch_roc
    scheduler_D.step()
    scheduler_A.step()
    scheduler_C.step()

    b=time.time()

    return total_loss / num_batches, total_acc / num_batches, total_f1 / num_batches, total_roc / num_batches, b-a



def Train(train_v_data,ce_f,Optimizers_D_v,Optimizers_C_v,Optimizers_A_v,Schedulers_D_v,Schedulers_C_v, Schedulers_A_v,all_folds_acc_v,all_folds_acc_t,all_folds_roc_v,all_folds_roc_t,es,k,val_f1,best_acc,best_roc,val_acc,best_val_acc,val_roc,path_,Break,model_idx,barrier,classes_num,scheduler_D,scheduler_C,scheduler_A,args,Discriminator, Classifier, Attention,training_data, validation_data,test_data, optimizer_D,optimizer_C,optimizer_A, criterion,bce, batch_size, device,Discriminators,Classifiers, Attentions):
    """
    The main training and validation loop
    At the end of each fold, results are saved to txt file
    """
    train_epoch_loss, train_epoch_acc, train_epoch_roc, train_epoch_f1 = [[] for _ in range(classes_num)],[[] for _ in range(classes_num)],[[] for _ in range(classes_num)],[[] for _ in range(classes_num)]
    train_v_epoch_loss, train_v_epoch_acc, train_v_epoch_roc, train_v_epoch_f1 = [],[],[],[]
    validation_epoch_loss, validation_epoch_acc,validation_epoch_roc, validation_epoch_f1 = [[] for _ in range(classes_num)],[[] for _ in range(classes_num)],[[] for _ in range(classes_num)],[[] for _ in range(classes_num)]
    validation_epoch_acc_B=[]
    epochs = args['epochs']
    episodes = args['episodes']
    best_acc.value=0.0
    best_roc.value=0.0

    bva = [[] for _ in range(classes_num)]

    for i in range(episodes):
        barrier.wait()  # Syncing models
        if Break.value==1:
            print("Process "+str(model_idx+1)+" is broken.")
            break
        else:
            print(f"{bcolors.RED}[INFO]: Model {model_idx+1} Episode {i+1} of {episodes}{bcolors.ENDC}")
            for j in range(epochs):


                epoch_loss, epoch_acc, epoch_f1, epoch_roc, sec = train_(scheduler_D,scheduler_C,scheduler_A,Discriminator, Classifier, Attention,training_data, optimizer_D,optimizer_C, optimizer_A, criterion,bce, batch_size, device)
                train_epoch_loss[model_idx].append(epoch_loss)
                train_epoch_acc[model_idx].append(epoch_acc)
                train_epoch_f1[model_idx].append(epoch_f1)
                train_epoch_roc[model_idx].append(epoch_roc)

                barrier.wait() # Syncing models

                print(f"[INFO]: Model {model_idx+1} Epoch {j+1} of {epochs}")
                print(f"{bcolors.YELLOW} Model {model_idx+1} Training loss: {epoch_loss:.3f}, training acc: {epoch_acc:.3f}, training f1: {epoch_f1:.3f}, training roc: {epoch_roc:.3f}, time in sec: {sec:.3f}{bcolors.ENDC}")

                last = barrier.wait() # Indicator of the remaining processes

                if last == 0 and j!=epochs-1:
                    epoch_acc_v, epoch_f1_v, epoch_roc_v, sec_v,Weights, P_Labels=val_test(Discriminators, Classifiers, Attentions, validation_data, batch_size, device,classes_num)

                    val_acc.append(epoch_acc_v)
                    val_f1.append(epoch_f1_v)
                    val_roc.append(epoch_roc_v)

                    print(f"{bcolors.GREEN}Validation acc: {epoch_acc_v:.3f}, validation f1: {epoch_f1_v:.3f}, validation roc: {epoch_roc_v}, time in sec: {sec_v:.3f}{bcolors.ENDC}")
                    epoch_acc_test, epoch_f1_test, epoch_roc_test, sec_test,Weights_test, P_Labels_test=val_test(Discriminators, Classifiers, Attentions, test_data, len(test_data), device,classes_num)
                    print(f"{bcolors.BLUE}Test acc: {epoch_acc_test:.3f}, test f1: {epoch_f1_test:.3f}, test roc: {epoch_roc_test}, time in sec: {sec_test:.3f}{bcolors.ENDC}")



                elif last == 0 and j==epochs-1 and args['epochs_v']:
                    epoch_acc_v, epoch_f1_v, epoch_roc_v, sec_v,Weights, P_Labels=val_test(Discriminators, Classifiers, Attentions, validation_data, batch_size, device,classes_num)

                    val_acc.append(epoch_acc_v)
                    val_f1.append(epoch_f1_v)
                    val_roc.append(epoch_roc_v)

                    print(f"{bcolors.GREEN}Validation acc: {epoch_acc_v:.3f}, validation f1: {epoch_f1_v:.3f}, validation roc: {epoch_roc_v}, time in sec: {sec_v:.3f}{bcolors.ENDC}")
                    """
                    training loop in validation style
                    """
                    for jv in range(args['epochs_v']):
                        epoch_v_loss, epoch_v_acc, epoch_v_f1, epoch_v_roc, sec_t_v = train_v(Discriminators, Classifiers, Attentions, train_v_data, batch_size, device,classes_num, ce_f,Optimizers_D_v,Optimizers_C_v,Optimizers_A_v,Schedulers_D_v,Schedulers_C_v, Schedulers_A_v)
                        train_v_epoch_loss.append(epoch_v_loss)
                        train_v_epoch_acc.append(epoch_v_acc)
                        train_v_epoch_f1.append(epoch_v_f1)
                        train_v_epoch_roc.append(epoch_v_roc)

                        print(f"{bcolors.ORANGE}Training _v loss: {epoch_v_loss:.3f},Training _v acc: {epoch_v_acc:.3f}, Training _v f1: {epoch_v_f1:.3f}, Training _v roc: {epoch_v_roc}, time in sec: {sec_t_v:.3f}{bcolors.ENDC}")

                        epoch_acc_v, epoch_f1_v, epoch_roc_v, sec_v,Weights, P_Labels=val_test(Discriminators, Classifiers, Attentions, validation_data, batch_size, device,classes_num)

                        val_acc.append(epoch_acc_v)
                        val_f1.append(epoch_f1_v)
                        val_roc.append(epoch_roc_v)

                        print(f"{bcolors.GREEN}Validation acc: {epoch_acc_v:.3f}, validation f1: {epoch_f1_v:.3f}, validation roc: {epoch_roc_v}, time in sec: {sec_v:.3f}{bcolors.ENDC}")

                        # Early stopping condition after the end of the last epoch
                        if jv==args['epochs_v']-1 :

                            if round(sum(val_acc[-(epochs+args['epochs_v']):])/len(val_acc[-(epochs+args['epochs_v']):]),3) > best_acc.value:
                                best_acc.value= round(sum(val_acc[-(epochs+args['epochs_v']):])/len(val_acc[-(epochs+args['epochs_v']):]),3)
                                best_roc.value= sum(val_roc[-epochs:])/len(val_roc[-epochs:])
                                print("best acc = ",best_acc.value)
                                print("best roc = ",[round(num, 3) for num in best_roc.value.tolist()])
                                es.value = 0
                                # Saving the best models, attention weights, predicted and target labels
                                for i in range(classes_num):
                                    torch.save(Discriminators[i],path_.value+"/Discriminator_k"+str(k+1)+"_"+str(i+1)+".pt")
                                    torch.save(Attentions[i],path_.value+"/Attention_k"+str(k+1)+"_"+str(i+1)+".pt")
                                    torch.save(Classifiers[i],path_.value+"/Classifier_k"+str(k+1)+"_"+str(i+1)+".pt")
                                torch.save(Weights,path_.value+"/Weights_"+str(k+1)+".pt")
                                torch.save(P_Labels,path_.value+"/P_Labels_"+str(k+1)+".pt")

                            else:
                                es.value += 1
                                print("Counter {} of 5".format(es.value))
                                if es.value > 5:
                                    best_val_acc.append(best_acc.value)
                                    all_folds_acc.append(best_acc.value)
                                    all_folds_roc.append([round(num, 3) for num in best_roc.value.tolist()])
                                    print("Early stopping with best_acc: ", best_acc.value, "and val_acc for this epoch: ", round(sum(val_acc[-epochs:])/len(val_acc[-epochs:]),3), "...")
                                    Break.value=1
                                    break

                elif last == 0 and j==epochs-1:

                    epoch_acc_v, epoch_f1_v, epoch_roc_v, sec_v,Weights, P_Labels=val_test(Discriminators, Classifiers, Attentions, validation_data, batch_size, device,classes_num)

                    val_acc.append(epoch_acc_v)
                    val_f1.append(epoch_f1_v)
                    val_roc.append(epoch_roc_v)

                    print(f"{bcolors.GREEN}Validation acc: {epoch_acc_v:.3f}, validation f1: {epoch_f1_v:.3f}, validation roc: {epoch_roc_v}, time in sec: {sec_v:.3f}{bcolors.ENDC}")
                    epoch_acc_test, epoch_f1_test, epoch_roc_test, sec_test,Weights_test, P_Labels_test=val_test(Discriminators, Classifiers, Attentions, test_data, len(test_data), device,classes_num)
                    print(f"{bcolors.BLUE}Test acc: {epoch_acc_test:.3f}, test f1: {epoch_f1_test:.3f}, test roc: {epoch_roc_test}, time in sec: {sec_test:.3f}{bcolors.ENDC}")
                    # Early stopping condition after the end of the last epoch

                    if round(sum(val_acc[-epochs:])/len(val_acc[-epochs:]),3) > best_acc.value:
                        best_acc.value= round(sum(val_acc[-epochs:])/len(val_acc[-epochs:]),3)
                        best_roc.value= sum(val_roc[-epochs:])/len(val_roc[-epochs:])
                        print("best acc = ",best_acc.value)
                        print("best roc = ",[round(num, 3) for num in best_roc.value.tolist()])
                        es.value = 0
                        # Saving the best models, attention weights, predicted and target labels
                        for i in range(classes_num):
                            torch.save(Discriminators[i],path_.value+"/Discriminator_k"+str(k+1)+"_"+str(i+1)+".pt")
                            torch.save(Attentions[i],path_.value+"/Attention_k"+str(k+1)+"_"+str(i+1)+".pt")
                            torch.save(Classifiers[i],path_.value+"/Classifier_k"+str(k+1)+"_"+str(i+1)+".pt")
                        torch.save(Weights,path_.value+"/Weights_"+str(k+1)+".pt")
                        torch.save(P_Labels,path_.value+"/P_Labels_"+str(k+1)+".pt")

                    else:
                        es.value += 1
                        print("Counter {} of 5".format(es.value))
                        if es.value > 5:
                            best_val_acc.append(best_acc.value)
                            all_folds_acc_v.append(best_acc.value)
                            all_folds_roc_v.append([round(num, 3) for num in best_roc.value.tolist()])
                            print("Early stopping with best_acc: ", best_acc.value, "and val_acc for this epoch: ", round(sum(val_acc[-epochs:])/len(val_acc[-epochs:]),3), "...")
                            Break.value=1
                            #test step after early stopping
                            #load bast saved models
                            L_Ds=[]
                            L_Cs=[]
                            L_As=[]
                            for i in range(classes_num):
                                L_Ds.append(torch.load(path_.value+"/Discriminator_k"+str(k+1)+"_"+str(i+1)+".pt"))
                                L_Cs.append(torch.load(path_.value+"/Classifier_k"+str(k+1)+"_"+str(i+1)+".pt"))
                                L_As.append(torch.load(path_.value+"/Attention_k"+str(k+1)+"_"+str(i+1)+".pt"))
                            epoch_acc_test, epoch_f1_test, epoch_roc_test, sec_test,Weights_test, P_Labels_test=val_test(L_Ds, L_Cs, L_As, test_data, len(test_data), device,classes_num)
                            torch.save(Weights_test,path_.value+"/Weights_test"+str(k+1)+".pt")
                            torch.save(P_Labels_test,path_.value+"/P_Labels_test"+str(k+1)+".pt")
                            all_folds_acc_t.append(epoch_acc_test)
                            all_folds_roc_t.append(epoch_roc_test)
                            print(f"{bcolors.BLUE}Test acc: {epoch_acc_test:.3f}, test f1: {epoch_f1_test:.3f}, test roc: {epoch_roc_test}, time in sec: {sec_test:.3f}{bcolors.ENDC}")
                            break

    f = open(path_.value+"/metrics_k"+str(k+1)+".txt", "a")
    f.write(str(args)+"\n"+"Model "+str(model_idx+1)+"\n"+"train_epoch_loss "+str(train_epoch_loss[model_idx])+"\n"+
    "train_epoch_acc "+str(train_epoch_acc[model_idx])+"\n"+
    "train_epoch_f1 "+str(train_epoch_f1[model_idx])+"\n"+
    "train_epoch_roc "+str(train_epoch_roc[model_idx])+"\n"+
    "train_v_epoch_loss "+str(train_v_epoch_loss)+"\n"+
    "train_v_epoch_acc "+str(train_v_epoch_acc)+"\n"+
    "train_v_epoch_f1 "+str(train_v_epoch_f1)+"\n"+
    "train_v_epoch_roc "+str(train_v_epoch_roc)+"\n"+
    "val_epoch_acc "+str(val_acc)+"\n"+
    "val_epoch_f1 "+str(val_f1)+"\n"+
    "val_epoch_roc "+str(val_roc)+"\n"+
    "best_validation_acc "+str(best_val_acc)+"\n"+
    "all_folds_acc_v "+str(all_folds_acc_v)+"\n"+
    "all_folds_roc_v "+str(all_folds_roc_v)+"\n"+
    "all_folds_acc_t "+str(all_folds_acc_t)+"\n"+
    "all_folds_roc_t "+str(all_folds_roc_t)+"\n")
    f.close()

def val_test(Disc, Clas, Att, val_data, batch_size, device,classes_num):
    """
    Validation using all models
    Returns all metrics and execution time
    """
    a=time.time()
    validation_dataloader = DataLoader(val_data, batch_size=batch_size)

    total_acc = 0.0
    total_f1 = 0.0
    total_roc = torch.tensor(0)

    num_batches = 0
    Weights=[[] for _ in range(len(Disc))]
    P_Labels = []

    for batch_idx, (x1,y) in enumerate(validation_dataloader):
        Y_ = []
        num_batches += 1
        for idx in range(len(Disc)):
            Disc[idx].to(device)
            Disc[idx].eval()
            Clas[idx].to(device)
            Clas[idx].eval()
            Att[idx].to(device)
            Att[idx].eval()

            z1=Disc[idx](x1.to(device))  # Discriminator forward pass
            attended1,weights = Att[idx](z1) # Attention forward pass
            za1=torch.add(z1,attended1) # Residual connection
            za1=nn.BatchNorm1d(za1.size(-2))(za1)
            Weights[idx].append(weights) # Saving weights
            y_ = Clas[idx](za1) # Classifier prediction
            Y_.append(y_)

        with torch.no_grad():

            YY_ = F.softmax(torch.cat(Y_, dim=1),dim=1) # Softmax on all models predictions

            One_hot_y_ = torch.eye(YY_.shape[1])[torch.argmax(YY_,dim = 1)] # One hot encoding prediction

            P_Labels.append((One_hot_y_,y)) # Saving labels


            acc = accuracy(torch.argmax(One_hot_y_,dim=1).to(device), torch.argmax(y,dim=1).to(device),'multiclass',num_classes=classes_num,average='weighted')
            total_acc+= acc.item()
            f1 = f1_score(torch.argmax(One_hot_y_,dim=1).to(device), torch.argmax(y,dim=1).to(device), task="multiclass", num_classes =classes_num, average='weighted' )
            total_f1 += f1.item()

            try:
                roc = torch.tensor(roc_auc_score(torch.argmax(y,dim=1).to(device), One_hot_y_.to(device),average=None, multi_class='ovr'))
            except ValueError:
                print("error in roc, not all classes in the batch")
                roc=torch.tensor([0.5 for i in range(classes_num)])
            total_roc = torch.add(roc,total_roc)

    b=time.time()
    return total_acc/num_batches,total_f1/num_batches, total_roc/num_batches, b-a,Weights, P_Labels

def train_v(Disc, Clas, Att, train_v_data, batch_size, device,classes_num, ce_f,Optimizers_D_v,Optimizers_C_v,Optimizers_A_v,Schedulers_D_v,Schedulers_C_v, Schedulers_A_v):
    """
    Training loop in validation style
    Returns all metrics and execution time
    """
    a=time.time()
    training_v_dataloader = DataLoader(train_v_data, batch_size=batch_size)

    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    total_roc = torch.tensor(0)

    num_batches = 0
    #Weights=[[] for _ in range(len(Disc))]
    #P_Labels = []

    for batch_idx, (x1,y) in enumerate(training_v_dataloader):
        Y_ = []
        num_batches += 1
        for idx in range(len(Disc)):
            Disc[idx].to(device)
            Disc[idx].train()
            Clas[idx].to(device)
            Clas[idx].train()
            Att[idx].to(device)
            Att[idx].train()

            z1=Disc[idx](x1.to(device))  # Discriminator forward pass
            attended1,weights = Att[idx](z1) # Attention forward pass
            za1=torch.add(z1,attended1) # Residual connection
            za1=nn.BatchNorm1d(za1.size(-2))(za1)

            y_ = Clas[idx](za1) # Classifier prediction
            Y_.append(y_)

        Y_=torch.cat(Y_, dim=1)
        Y_.requires_grad_(True)
        YY_ = F.softmax(Y_,dim=1) # Softmax on all models predictions
        YY_.requires_grad_(True)
        One_hot_y_ = torch.eye(YY_.shape[1])[torch.argmax(YY_,dim = 1)] # One hot encoding prediction
        One_hot_y_.requires_grad_(True)
        loss = ce_f(YY_.to(device),y.to(device))
        total_loss+= loss.item()
        with torch.no_grad():

            acc = accuracy(torch.argmax(One_hot_y_,dim=1).to(device), torch.argmax(y,dim=1).to(device),'multiclass',num_classes=classes_num,average='weighted')
            total_acc+= acc.item()
            f1 = f1_score(torch.argmax(One_hot_y_,dim=1).to(device), torch.argmax(y,dim=1).to(device), task="multiclass", num_classes =classes_num, average='weighted' )
            total_f1 += f1.item()
            try:
                roc = torch.tensor(roc_auc_score(torch.argmax(y,dim=1).to(device), One_hot_y_.to(device),average=None, multi_class='ovr'))
            except ValueError:
                print("error in roc, not all classes in the batch")
                roc=torch.tensor([0.5 for i in range(classes_num)])
            total_roc = torch.add(roc,total_roc)

        for idx in range(len(Disc)):
            Optimizers_D_v[idx].zero_grad()
            Optimizers_C_v[idx].zero_grad()
            Optimizers_A_v[idx].zero_grad()

        loss.backward()

        for idx in range(len(Disc)):
            Optimizers_D_v[idx].step()
            Optimizers_C_v[idx].step()
            Optimizers_A_v[idx].step()
            Schedulers_D_v[idx].step()
            Schedulers_C_v[idx].step()
            Schedulers_A_v[idx].step()


    b=time.time()
    return total_loss/num_batches, total_acc/num_batches,total_f1/num_batches, total_roc/num_batches, b-a


def parallel_training(train_v_data, ce_f, Optimizers_D_v, Optimizers_C_v, Optimizers_A_v, Schedulers_D_v, Schedulers_C_v, Schedulers_A_v, all_folds_acc_v,all_folds_acc_t,all_folds_roc_v,all_folds_roc_t,es,k,val_f1,best_acc,best_roc,val_acc,best_val_acc,val_roc,Break,path_,model_idx,barrier,Schedulers_D,Schedulers_C,Schedulers_A,Discriminators,Classifiers,Attentions,Training_data,Val_data,Test_data,Optimizers_D,Optimizers_C,Optimizers_A,args, criterion,BCE, batch_size, device,classes_num):
    scheduler_D = Schedulers_D[model_idx]
    scheduler_C = Schedulers_C[model_idx]
    scheduler_A = Schedulers_A[model_idx]
    Discriminator = Discriminators[model_idx]
    Classifier = Classifiers[model_idx]
    Attention = Attentions[model_idx]
    training_data = Training_data[model_idx]
    validation_data = Val_data[model_idx]
    test_data = Test_data[model_idx]
    optimizer_D = Optimizers_D[model_idx]
    optimizer_C = Optimizers_C[model_idx]
    optimizer_A = Optimizers_A[model_idx]
    bce=BCE[model_idx]

    Train(train_v_data,ce_f,Optimizers_D_v,Optimizers_C_v,Optimizers_A_v,Schedulers_D_v,Schedulers_C_v, Schedulers_A_v,all_folds_acc_v,all_folds_acc_t,all_folds_roc_v,all_folds_roc_t,es,k,val_f1,best_acc,best_roc,val_acc,best_val_acc,val_roc,path_,Break,model_idx,barrier,classes_num,scheduler_D,scheduler_C,scheduler_A,args,Discriminator, Classifier, Attention,training_data, validation_data, test_data, optimizer_D,optimizer_C,optimizer_A, criterion,bce, batch_size, device,Discriminators,Classifiers, Attentions)



if __name__ == '__main__':

    randomseed = 1
    np.random.seed(randomseed)
    torch.manual_seed(randomseed)
    torch.cuda.manual_seed(randomseed)
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser()
    parser.add_argument('-da', '--dataset', type=int, default=3,
        help='Dataset id')
    parser.add_argument('-E', '--episodes', type=int, default=25,
        help='number of episodes , outter loop')
    parser.add_argument('-e', '--epochs', type=int, default=10,
        help='number of epochs , inner loop')
    parser.add_argument('-ev', '--epochs_v', type=int,
        help='number of epochs for training in validation style, leave empty if not used')
    parser.add_argument('-lrd', '--learning_rated', type=float, default=1e-3,
        help='Learning rate for discriminator network')
    parser.add_argument('-lrc', '--learning_ratec', type=float, default=1e-3,
        help='Learning rate for classifier network')
    parser.add_argument('-lra', '--learning_ratea', type=float, default=1e-3,
        help='Learning rate for att network')
    parser.add_argument('-lrv', '--learning_rate_v', type=float, default=1e-4,
        help='Learning rate for training in validation style')
    parser.add_argument('-drd', '--drop_rated', type=float, default=0.,
        help='dropout rate for discriminator network')
    parser.add_argument('-drc', '--drop_ratec', type=float, default=0.,
        help='dropout rate for classifier network')
    parser.add_argument('-n', '--n_pairs', type=int, default=1,
        help='number of pairs')
    parser.add_argument('-m', '--margin', type=float, default=1e-3,
        help='margin for cos similarity')
    parser.add_argument('-t', '--threshold', type=float, default=0.,
        help='Variance threshold')
    parser.add_argument('-b', '--batch_size', type=int, default=128,
        help='batch size')
    parser.add_argument('-d', '--device', type=str, default='cuda',
        help='device to use, cuda or cpu')
    parser.add_argument('-K', '--Kfolds', type=int, default=5,
        help='number of validdation folds')
    parser.add_argument('-p', '--path', type=str,
        help='dataset file path')
    parser.add_argument('-c', '--datasetnames', nargs='+',
        help='dataset attributes names')
    parser.add_argument('-cn', '--classes_num', type=int,
        help='number of classes')
    parser.add_argument('-do', '--discout', type=int, default=128,
        help='discriminator output size')

    args = vars(parser.parse_args())

    if args['path']:
        # If dataset is from a file
        manager=Manager()
        classes_num = args['classes_num']
        F, T, input_size, B = get_dataset_multi_from_file(args['path'],classes_num,args['threshold'] )
        F_m = torch.cat(F)
        T_m = torch.cat(T)

        date=time.ctime()

        path="runs_folds/"+str(date).replace(" ", "_").replace(":", "_")+"multi_attention_"+str(args['path'])
        os.mkdir(path)

        path_ = manager.Value(c_char_p,"runs_episodes/"+str(date).replace(" ", "_").replace(":", "_")+"multi_attention_"+str(args['path']))
        os.mkdir(path_.value)
    else:
        # If dataset is from OpenMl
        F, T, input_size, B = get_dataset_multi(get_dataset_id_multi(args['dataset'])[0],get_dataset_id_multi(args['dataset'])[1],args['threshold'])
        print("input_size : ",input_size )
        F_m = torch.cat(F)
        T_m = torch.cat(T)
        manager=Manager()
        classes_num = get_dataset_id_multi(args['dataset'])[1]

        date=time.ctime()

        path="runs_folds/"+str(date).replace(" ", "_").replace(":", "_")+"multi_attention_"+str(args['dataset'])
        os.mkdir(path)

        path_ = manager.Value(c_char_p,"runs_episodes/"+str(date).replace(" ", "_").replace(":", "_")+"multi_attention_"+str(args['dataset']))
        os.mkdir(path_.value)

    learning_rated = args['learning_rated']
    learning_ratec = args['learning_ratec']
    lra = args['learning_ratea']
    learning_rate_v = args['learning_rate_v']
    drop_rated = args['drop_rated']
    drop_ratec = args['drop_ratec']
    discout = args['discout']
    epochs = args['epochs']
    episodes = args['episodes']
    patience = 5 # 5 episodes == 50 epochs
    n_pairs = args['n_pairs']
    margin = args['margin']
    batch_size = args['batch_size']
    device = args['device']
    K = args['Kfolds']


    args_ =args

    criterion = Cos_loss(margin)

    stratified_kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=randomseed)

    max_training_time = 3600

    Fold_train_loss,Fold_val_loss,Fold_train_acc,Fold_val_acc,Fold_train_f1,Fold_val_f1,Fold_train_roc,Fold_val_roc=[],[],[],[],[],[],[],[]
    bvr=[]


    all_folds_acc_v  = Manager().list()
    all_folds_roc_v = Manager().list()
    all_folds_acc_t  = Manager().list()
    all_folds_roc_t = Manager().list()

    start_time = time.time()

    # Saving the used arguments in a txt file
    f = open(path_.value+"/args.txt", "a")
    f.write(str(args))
    f.close()

    for k, (train_index, test_index) in enumerate(stratified_kfold.split(F_m, torch.argmax(T_m, dim=1))):
        print(f"{bcolors.CYAN}[INFO]: Fold {k+1} of {K}{bcolors.ENDC}")


        K_=k
        Training_data = []
        Val_data = []
        Test_data = []
        Discriminators = []
        Classifiers = []
        Attentions = []
        Optimizers_D = []
        Optimizers_C = []
        Optimizers_A = []
        Schedulers_D = []
        Schedulers_C = []
        Schedulers_A = []
        BCE = []
        # for training in validation style
        Optimizers_D_v = []
        Optimizers_C_v = []
        Optimizers_A_v = []
        Schedulers_D_v = []
        Schedulers_C_v = []
        Schedulers_A_v = []



        for class_id in range(classes_num):
            features_test, targets_test ,b_test= split_tensors(T_m[test_index], F_m[test_index]) #test data
            F_tr,F_va,T_tr,T_va=train_test_split(F_m[train_index],T_m[train_index],test_size=0.1,
                                                    shuffle=True, random_state=1,
                                                    stratify= torch.argmax(T_m[train_index], dim=1)) #spliting training data to train and validation
            features_train, targets_train, b_train = split_tensors(T_tr, F_tr)
            features_val, targets_val, b_val = split_tensors(T_va, F_va)

            class_weights = compute_class_weight('balanced', classes=np.array(torch.unique(torch.argmax(torch.cat(targets_train),dim=1))), y=np.array(torch.argmax(torch.cat(targets_train),dim=1)))
            class_weights=torch.tensor(class_weights, dtype=torch.float32)
            print("all classes weights : ",class_weights)
            pos_weights = (torch.sum(class_weights)-class_weights[class_id]) /class_weights[class_id]
            print("class : ", class_id+1," pos weight :",pos_weights)
            bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            BCE.append(bce)
            ce_f = nn.CrossEntropyLoss(weight=class_weights)

            train_v_data = openmldataset_val_multi(features_train, targets_train)

            training_data = openmldataset_tt_multi(features_train, targets_train, b_train, n_pairs, classes_num, class_id)
            validation_data = openmldataset_val_multi(features_val, targets_val)
            test_data = openmldataset_val_multi(features_test, targets_test)

            Discriminator = DiscNN(input_size,drop_rated,discout)
            Classifier = ClassifierNN(discout,input_size,drop_ratec)
            Attention = AttentionNN(discout)

            optimizer_D = torch.optim.Adam(Discriminator.parameters(), lr=learning_rated)
            optimizer_C = torch.optim.Adam(Classifier.parameters(), lr=learning_ratec)
            optimizer_A = torch.optim.Adam(Attention.parameters(), lr=lra)

            scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.9)
            scheduler_C = torch.optim.lr_scheduler.ExponentialLR(optimizer_C, gamma=0.9)
            scheduler_A = torch.optim.lr_scheduler.ExponentialLR(optimizer_A, gamma=0.9)

            optimizer_D_v = torch.optim.Adam(Discriminator.parameters(), lr=learning_rate_v)
            optimizer_C_v = torch.optim.Adam(Classifier.parameters(), lr=learning_rate_v)
            optimizer_A_v = torch.optim.Adam(Attention.parameters(), lr=learning_rate_v)

            scheduler_D_v = torch.optim.lr_scheduler.ExponentialLR(optimizer_D_v, gamma=0.9)
            scheduler_C_v = torch.optim.lr_scheduler.ExponentialLR(optimizer_C_v, gamma=0.9)
            scheduler_A_v = torch.optim.lr_scheduler.ExponentialLR(optimizer_A_v, gamma=0.9)

            Training_data.append(training_data)
            Val_data.append(validation_data)
            Test_data.append(test_data)

            Discriminators.append(Discriminator)
            Classifiers.append(Classifier)
            Attentions.append(Attention)

            Optimizers_D.append(optimizer_D)
            Optimizers_C.append(optimizer_C)
            Optimizers_A.append(optimizer_A)

            Schedulers_D.append(scheduler_D)
            Schedulers_C.append(scheduler_C)
            Schedulers_A.append(scheduler_A)

            Optimizers_D_v.append(optimizer_D_v)
            Optimizers_C_v.append(optimizer_C_v)
            Optimizers_A_v.append(optimizer_A_v)
            Schedulers_D_v.append(scheduler_D_v)
            Schedulers_C_v.append(scheduler_C_v)
            Schedulers_A_v.append(scheduler_A_v)

        model_indices = list(range(classes_num))

        processes = []
        barrier = Barrier(classes_num) # A barrier to sync all processes
        Break = manager.Value('i',0)
        last_process = Value('i', -1)
        val_acc = Manager().list()
        val_f1  = Manager().list()
        best_val_acc = Manager().list()
        val_roc = Manager().list()
        best_acc = manager.Value('f',0.0)
        best_roc = manager.Value('f',0.0)
        es = manager.Value('i',0) # Early stopping indicator
        for model_idx in model_indices:
            p= Process(target=parallel_training,args=(train_v_data, ce_f, Optimizers_D_v, Optimizers_C_v, Optimizers_A_v, Schedulers_D_v, Schedulers_C_v, Schedulers_A_v,all_folds_acc_v,all_folds_acc_t,all_folds_roc_v,all_folds_roc_t,es,k,val_f1,best_acc,best_roc,val_acc,best_val_acc,val_roc,Break,path_,model_idx,barrier,Schedulers_D,Schedulers_C,Schedulers_A,Discriminators,Classifiers,Attentions,Training_data,Val_data,Test_data,Optimizers_D,Optimizers_C,Optimizers_A, args, criterion,BCE, batch_size, device, classes_num))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

        elapsed_time = time.time() - start_time
        if elapsed_time >= max_training_time:
            user_input = input("Maximum training time reached. Do you want to stop training? (y/n): ")
            if user_input.lower() == 'y':
                print("Training stopped.")
                break
            else:
                start_time = time.time()

    print(f"{bcolors.CYAN}training complete{bcolors.ENDC}")
