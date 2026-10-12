import torch
import openml
from openml import tasks, runs
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,StandardScaler
from torch.utils.data import Dataset
import numpy as np
import category_encoders as ce
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

torch.set_default_dtype(torch.float32)

def get_dataset_id(task_id):
    """
    Retreiving dataset id using task id, binary classification
    """
    benchmark = openml.study.get_suite('OpenML-CC18')
    task_frame=tasks.list_tasks(output_format="dataframe", task_id=benchmark.tasks)
    return int(task_frame.did[task_id])

def get_dataset_id_multi(task_id):
    """
    Retreiving dataset id and number of classes using task id, multiclass classification
    """
    benchmark = openml.study.get_suite('AutoML-Benchmark')
    task_frame=tasks.list_tasks(output_format="dataframe", task_id=benchmark.tasks)
    return int(task_frame.did[task_id]), int(task_frame.NumberOfClasses[task_id])

def get_dataset(dataset_id):
    """
    Data donwload and preprocessing, for binary classification
    """
    data_frame, a, b, c = openml.datasets.get_dataset(dataset_id).get_data(dataset_format="dataframe")
    item_n=0
    for item in c:
        if data_frame[item].isna().all() or (data_frame[item] == 'None').all():
            data_fram = data_frame.drop(columns=[item],inplace=True)
            del b[item_n]
        item_n +=1
    data_frame = data_frame.dropna()
    targets = OneHotEncoder(drop='if_binary').fit_transform(data_frame[[data_frame.columns[-1]]]).toarray()
    index_n=[i for i, x in enumerate(b) if not x]
    index_c=[i for i, x in enumerate(b) if x]

    cat=False
    num=False
    if index_c!=[]:
        del index_c[-1]
        categ=pd.concat([data_frame[data_frame.columns[index_c]]])
        encoder = ce.OrdinalEncoder()
        features_c = encoder.fit_transform(categ,targets).values
        cat=True
    if index_n!=[] and cat==False:
        del index_n[-1]
        numi=pd.concat([data_frame[data_frame.columns[index_n]]])
        scaler = StandardScaler()
        features_n =scaler.fit_transform(numi)
        num=True
    elif index_n!=[] and cat==True:
        numi=pd.concat([data_frame[data_frame.columns[index_n]]])
        scaler = StandardScaler()
        features_n =scaler.fit_transform(numi)
        num=True
    if num==True and cat==True:
        features = np.concatenate((features_n, features_c), axis=1)
    elif num==True and cat==False:
        features = features_n
    elif num==False and cat==True:
        features = features_c

    features_tensor = torch.tensor(features, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    indices = torch.randperm(features_tensor.size()[0])
    features_tensor=features_tensor[indices]
    targets_tensor=targets_tensor[indices]
    return features_tensor, targets_tensor, len(b)-1

def creat_folds(features, targets):
    """
    Not used
    """
    indices = torch.randperm(features.size()[0])
    features=features[indices]
    targets=targets[indices]
    gap = features.size()[0]//5
    for i in range(5):
        features_v = features[i*gap:(i+1)*gap]
        targets_v = targets[i*gap:i*gap]
        if i==0:
            featues_t = features[i+1*gap:]
            targets_t = targets[i+1*gap:]
        elif i==4:
            featues_t = features[:i*gap]
            targets_t = targets[:i*gap]
        else:
            featues_t = torch.cat((features[0:(i)*gap], features[((i+1)*gap):]),0)
            targets_t = torch.cat((targets[0:(i)*gap], targets[((i+1)*gap):]),0)


def get_dataset_multi(dataset_id,classes,threshold):
    """
    Data donwload and preprocessing, for multiclass classification
    Returns Features, Targets, Input Size, and size of each class
    """

    data_frame, a, b, c = openml.datasets.get_dataset(dataset_id).get_data(dataset_format="dataframe")
    data_frame = data_frame.dropna()
    index=-1
    if 'Class' in data_frame.columns:
        index = data_frame.columns.get_loc('Class')
        targets = OneHotEncoder(drop='if_binary').fit_transform(data_frame[['Class']]).toarray()
    elif 'class' in data_frame.columns:
        index = data_frame.columns.get_loc('class')
        targets = OneHotEncoder(drop='if_binary').fit_transform(data_frame[['class']]).toarray()
    index_n=[i for i, x in enumerate(b) if not x]
    index_c=[i for i, x in enumerate(b) if x]
    if index!=-1:
        index_c.remove(index)
    cat=False
    num=False
    if index_c!=[]:
        categ=pd.concat([data_frame[data_frame.columns[index_c]]])
        encoder = ce.OrdinalEncoder()
        features_c = encoder.fit_transform(categ,targets.argmax(axis=1)).values

        # Features reduction based on a variance threshold
        selector = VarianceThreshold(threshold)
        selector.fit(features_c)
        features_c=selector.transform(features_c)

        scaler_c = StandardScaler()
        features_c =scaler_c.fit_transform(features_c)
        cat=True
    if index_n!=[]:
        numi=pd.concat([data_frame[data_frame.columns[index_n]]])

        # Features reduction based on a variance threshold
        selector = VarianceThreshold(threshold)
        selector.fit(numi)
        numi=selector.transform(numi)

        scaler = StandardScaler()
        features_n =scaler.fit_transform(numi)
        num=True
    if num==True and cat==True:
        features = np.concatenate((features_n, features_c), axis=1)
    elif num==True and cat==False:
        features = features_n
    elif num==False and cat==True:
        features = features_c

    F=[]
    T=[]
    B=[]
    features_tensor = torch.tensor(features, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    one_hot = torch.eye(classes)
    for i in range(classes):
        indices = torch.nonzero(torch.all(targets_tensor == one_hot[i], dim=1))
        T.append(targets_tensor[indices].squeeze())
        F.append(features_tensor[indices].squeeze())
        B.append(len(indices))

    return F, T,F[0].size()[1], B

def get_dataset_multi_from_file(path,classes,threshold):
    """
    Data preprocessing, for multi class classification
    Returns Features, Targets, Input Size, and size of each class
    """

    data_frame = pd.read_csv(path)
    b=[]
    for dtype in data_frame.dtypes:
        if dtype == "object":
            b.append(True)
        else:
            b.append(False)
    data_frame = data_frame.dropna()
    index=-1
    if 'Class' in data_frame.columns:
        index = data_frame.columns.get_loc('Class')
        targets = OneHotEncoder(drop='if_binary').fit_transform(data_frame[['Class']]).toarray()
    elif 'class' in data_frame.columns:
        index = data_frame.columns.get_loc('class')
        targets = OneHotEncoder(drop='if_binary').fit_transform(data_frame[['class']]).toarray()
    index_n=[i for i, x in enumerate(b) if not x]
    index_c=[i for i, x in enumerate(b) if x]
    if index!=-1:
        index_c.remove(index)
    cat=False
    num=False
    if index_c!=[]:
        categ=pd.concat([data_frame[data_frame.columns[index_c]]])
        encoder = ce.OrdinalEncoder()
        features_c = encoder.fit_transform(categ,targets.argmax(axis=1)).values

        # Features reduction based on a variance threshold
        selector = VarianceThreshold(threshold)
        selector.fit(features_c)
        features_c=selector.transform(features_c)

        scaler_c = StandardScaler()
        features_c =scaler_c.fit_transform(features_c)
        cat=True
    if index_n!=[]:
        numi=pd.concat([data_frame[data_frame.columns[index_n]]])

        # Features reduction based on a variance threshold
        selector = VarianceThreshold(threshold)
        selector.fit(numi)
        numi=selector.transform(numi)

        scaler = StandardScaler()
        features_n =scaler.fit_transform(numi)
        num=True
    if num==True and cat==True:
        features = np.concatenate((features_n, features_c), axis=1)
    elif num==True and cat==False:
        features = features_n
    elif num==False and cat==True:
        features = features_c

    F=[]
    T=[]
    B=[]
    features_tensor = torch.tensor(features, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    one_hot = torch.eye(classes)
    for i in range(classes):
        indices = torch.nonzero(torch.all(targets_tensor == one_hot[i], dim=1))
        T.append(targets_tensor[indices].squeeze())
        F.append(features_tensor[indices].squeeze())
        B.append(len(indices))

    return F, T,F[0].size()[1], B


def split_tensors(T_m, F_m):
    """
    Spliting features and target tensors by classes
    Returns the splited features , targets tensors and their size
    """
    unique_values = torch.unique(T_m, dim=0)
    split_tensors = []
    split_tensors_f = []
    b=[]
    for value in unique_values.flip(0):
        split_tensor = T_m[T_m.eq(value).all(dim=1)]
        split_tensor_f = F_m[T_m.eq(value).all(dim=1)]
        split_tensors.append(split_tensor)
        split_tensors_f.append(split_tensor_f)
        b.append(split_tensor_f.size()[0])
    return split_tensors_f, split_tensors, b
