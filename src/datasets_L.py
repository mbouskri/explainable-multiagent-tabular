import torch
import openml
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch.utils.data import Dataset
import numpy as np
import category_encoders as ce

torch.set_default_dtype(torch.float32)

class openmldataset_toy(Dataset):
    def __init__(self, count, variables, n_pairs):

        self.features_tensor = torch.rand(count,variables, dtype=torch.float32)
        self.targets_tensor =(torch.rand(size=(count,1)) < 0.50).float()
        self.n = n_pairs

    def __len__(self):
        return len(self.targets_tensor)

    def __getitem__(self, idx):
        L=[]
        x1 = self.features_tensor[idx]
        y = self.targets_tensor[idx]
        r= list(range(0, idx)) + list(range(idx+1, len(self.targets_tensor)-1))
        index = np.random.choice(r, self.n, replace=False)
        X2 = self.features_tensor[index]
        for y_ in self.targets_tensor[index]:
            if y==y_:
                L.append(1)
            else:
                L.append(-1)
        L=torch.tensor(L, dtype=torch.float32)
        return x1.reshape(1,len(x1)), X2, L, y.reshape(1,1)

class openmldataset_new(Dataset):
    def __init__(self, dataset_id, n_pairs, is_numerical=False):
        self.data_frame, self._, self._, self._ = openml.datasets.get_dataset(dataset_id).get_data(dataset_format="dataframe")
        self.data_frame = self.data_frame.dropna()
        self.targets = OneHotEncoder(drop='if_binary').fit_transform(self.data_frame[[self.data_frame.columns[-1]]]).toarray()
        if is_numerical==False:
            encoder = ce.OrdinalEncoder()
            self.features = encoder.fit_transform(self.data_frame[self.data_frame.columns[:-1]],self.targets).values
        else:
            scaler = MinMaxScaler()
            self.features =scaler.fit_transform(self.data_frame[self.data_frame.columns[:-1]])

        self.features_tensor = torch.tensor(self.features, dtype=torch.float32)
        self.targets_tensor = torch.tensor(self.targets, dtype=torch.float32)
        indices = torch.randperm(self.features_tensor.size()[0])
        self.features_tensor=self.features_tensor[indices]
        self.targets_tensor=self.targets_tensor[indices]
        self.n = n_pairs

    def __len__(self):
        return len(self.targets_tensor)

    def __getitem__(self, idx):
        L=[]
        x1 = self.features_tensor[idx]
        y = self.targets_tensor[idx]
        r= list(range(0, idx)) + list(range(idx+1, len(self.targets_tensor)-1))
        index = np.random.choice(r, self.n, replace=False)
        X2 = self.features_tensor[index]
        for y_ in self.targets_tensor[index]:
            if y==y_:
                L.append(1)
            else:
                L.append(-1)
        L=torch.tensor(L, dtype=torch.float32)
        return x1.reshape(1,len(x1)), X2, L, y.reshape(1,1)



class openmldataset_tt_multi(Dataset):
    """
    Dataloader for training
    """
    def __init__(self, F,T, B, n_pairs,classes_num, class_id):
        self.features = F
        self.targets = T
        self.B = B
        self.n = n_pairs
        self.classes_num = classes_num
        self.c_id=class_id
        self.features_p = self.features[class_id]
        del self.features[class_id]
        self.features_n = torch.concatenate((self.features), dim=0)
        self.targets_p = self.targets[class_id]
        del self.targets[class_id]
        self.targets_n = torch.concatenate((self.targets), dim=0)
        self.targets_p_b = torch.ones(self.B[class_id])
        del self.B[class_id]
        self.targets_n_b = torch.zeros(sum(self.B))

        self.all_features = torch.concatenate((self.features_p,self.features_n), dim=0)
        self.all_targets = torch.concatenate((self.targets_p,self.targets_n), dim=0)
        self.all_targets_b = torch.concatenate((self.targets_p_b,self.targets_n_b), dim=0)
        indices = torch.randperm(self.all_features.size()[0])
        self.all_features = self.all_features[indices]
        self.all_targets = self.all_targets[indices]
        self.all_targets_b = self.all_targets_b[indices]

    def __len__(self):
        return len(self.all_targets)

    def __getitem__(self, idx):
        L=[]
        x1 = self.all_features[idx]
        y_b = self.all_targets_b[idx]
        y = self.all_targets[idx]
        r= list(range(0, idx)) + list(range(idx+1, len(self.all_targets)-1))
        index = np.random.choice(r, self.n, replace=False)
        X2 = self.all_features[index]
        for y_ in self.all_targets_b[index]:
            if y_b==y_:
                L.append(1)
            else:
                L.append(-1)
        L=torch.tensor(L, dtype=torch.float32)

        return x1.reshape(len(x1),1), X2.unsqueeze(-1), L, y_b.reshape(1,1), y

class openmldataset_val_multi(Dataset):
    """
    Dataloader for validation
    """
    def __init__(self, F,T):
        self.features = F
        self.targets = T

        self.all_features = torch.concatenate((self.features), dim=0)
        self.all_targets = torch.concatenate((self.targets), dim=0)

        indices = torch.randperm(self.all_features.size()[0])
        self.all_features = self.all_features[indices]
        self.all_targets = self.all_targets[indices]

    def __len__(self):
        return len(self.all_targets)

    def __getitem__(self, idx):

        x1 = self.all_features[idx]

        y = self.all_targets[idx]

        return x1.reshape(len(x1),1), y
