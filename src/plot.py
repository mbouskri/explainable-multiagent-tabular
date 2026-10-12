import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openml
from openml import tasks
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from torchmetrics.functional import accuracy

def get_classes_labels(dataset_id,classes):
    data_frame, a, b, c = openml.datasets.get_dataset(dataset_id).get_data(dataset_format="dataframe")
    data_frame = data_frame.dropna()
    index=-1
    if 'Class' in data_frame.columns:
        encoder = OneHotEncoder(drop='if_binary')
        index = data_frame.columns.get_loc('Class')
        del c[index]
        targets = encoder.fit_transform(data_frame[['Class']]).toarray()
        return encoder.get_feature_names_out(['Class']), c
    elif 'class' in data_frame.columns:
        encoder = OneHotEncoder(drop='if_binary')
        index = data_frame.columns.get_loc('class')
        del c[index]
        targets = encoder.fit_transform(data_frame[['class']]).toarray()
        return encoder.get_feature_names_out(['class']),c

def get_classes_labels_csv(path,classes):
    #data_frame, a, b, c = openml.datasets.get_dataset(dataset_id).get_data(dataset_format="dataframe")
    data_frame = pd.read_csv(path)

    data_frame = data_frame.dropna()
    c=data_frame.columns.to_list()
    index=-1
    if 'Class' in data_frame.columns:
        encoder = OneHotEncoder(drop='if_binary')
        index = data_frame.columns.get_loc('Class')
        del c[index]
        targets = encoder.fit_transform(data_frame[['Class']]).toarray()
        return encoder.get_feature_names_out(['Class']), c
    elif 'class' in data_frame.columns:
        encoder = OneHotEncoder(drop='if_binary')
        index = data_frame.columns.get_loc('class')
        del c[index]
        targets = encoder.fit_transform(data_frame[['class']]).toarray()
        print(encoder.categories_)
        return encoder.get_feature_names_out(['class']),c

def get_dataset_id_multi(task_id):
    benchmark = openml.study.get_suite('AutoML-Benchmark')
    task_frame=tasks.list_tasks(output_format="dataframe", task_id=benchmark.tasks)
    return int(task_frame.did[task_id]), int(task_frame.NumberOfClasses[task_id])

def get_classes_labels_csv(path,classes):
    #data_frame, a, b, c = openml.datasets.get_dataset(dataset_id).get_data(dataset_format="dataframe")
    data_frame = pd.read_csv(path)

    data_frame = data_frame.dropna()
    c=data_frame.columns.to_list()
    index=-1
    if 'Class' in data_frame.columns:
        encoder = OneHotEncoder(drop='if_binary')
        index = data_frame.columns.get_loc('Class')
        del c[index]
        targets = encoder.fit_transform(data_frame[['Class']]).toarray()
        return encoder.get_feature_names_out(['Class']), c
    elif 'class' in data_frame.columns:
        encoder = OneHotEncoder(drop='if_binary')
        index = data_frame.columns.get_loc('class')
        del c[index]
        targets = encoder.fit_transform(data_frame[['class']]).toarray()
        #print(encoder.categories_)
        return encoder.get_feature_names_out(['class']),c

def acc_all_folds(folder,num_classes):
    all_folds= []
    #stacked_array= [[] for i in range(num_classes)]
    for fold in range(10):
        acc=0
        W=torch.load(r"\runs_episodes\\" +folder +"\\Weights_"+str(fold+1)+".pt")
        P=torch.load(r"\runs_episodes\\" +folder +"\\P_Labels_"+str(fold+1)+".pt")
        for i in range(len(P)):
            acc += round(accuracy(torch.argmax(P[i][0],dim=1),torch.argmax(P[i][1],dim=1),'multiclass',num_classes=num_classes,average='weighted').item(),3)

        all_folds.append(round(acc/len(P),3))
    #for j in range(num_classes):
        #stacked_array[j].append(np.concatenate([all_folds[i][0][j] for i in range(10)]))
    return all_folds

def auc_all_folds(folder,num_classes,folds=3):
    all_folds= []
    #stacked_array= [[] for i in range(num_classes)]
    for fold in range(folds):
        total_roc=torch.tensor(0)
        W=torch.load(r"\runs_episodes\\" +folder +"\\Weights_"+str(fold+1)+".pt")
        P=torch.load(r"\runs_episodes\\" +folder +"\\P_Labels_"+str(fold+1)+".pt")
        for i in range(len(P)):
            try:
                roc = torch.tensor(roc_auc_score(P[i][1], P[i][0],average=None, multi_class='ovr'))
            except ValueError:
                print("error in roc, not all classes in the batch")
                roc=torch.tensor([0.5 for i in range(num_classes)])
            total_roc = torch.add(roc,total_roc)

        all_folds.append((total_roc/len(P)).tolist())
    #for j in range(num_classes):
        #stacked_array[j].append(np.concatenate([all_folds[i][0][j] for i in range(10)]))
    return torch.tensor(all_folds)

def Plot_AUC(folder,task,is_openml,file_name,num_classes=0,ylimit=(0.3,1.05),rot=False):
    if is_openml:
        task_id,num_classes = get_dataset_id_multi(task)
        y_names,c=get_classes_labels(get_dataset_id_multi(task)[0],get_dataset_id_multi(task)[1])
        all_folds_auc=auc_all_folds(folder,num_classes)

    else:
        y_names,c=get_classes_labels_csv(task,num_classes)
        all_folds_auc=auc_all_folds(folder,num_classes)
    tensor = np.array(all_folds_auc)
    plt.boxplot(tensor, showmeans=True, meanline=True, patch_artist=True,
        boxprops=dict(facecolor='white', color='black'),
        capprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        flierprops=dict(color='black', markeredgecolor='black'),
        medianprops=dict(color='red'),
        meanprops=dict(linestyle='-',color='blue')
        )
    median_legend = plt.Line2D([0], [0], color='red', label='Median', linestyle='-')
    mean_legend = plt.Line2D([0], [0], color='blue', label='Mean', linestyle='-')
    plt.ylabel('ROC_AUC Score')
    plt.xlabel('Classes')
    plt.ylim(ylimit)
    if rot:
        plt.xticks(range(1, num_classes + 1),y_names,rotation='vertical')
    else:
        plt.xticks(range(1, num_classes + 1),y_names)

    plt.legend(handles=[median_legend, mean_legend], loc='lower right')
    plt.savefig(file_name)

def plot_stacked(stacked_array, num_classes, task_id,file_name):

    y_names,c=get_classes_labels(get_dataset_id_multi(task_id)[0],get_dataset_id_multi(task_id)[1])
    axis_names = c
    fig, axes = plt.subplots(num_classes, 1, figsize=(8, 6))
    vmin = float('inf')  # Initialize the minimum value
    vmax = float('-inf')  # Initialize the maximum value

    for k in range(num_classes):
        heatmap = stacked_array[k][0].mean(axis=0)
        ax = axes[k]
        im = ax.imshow(heatmap, cmap='Blues')
        ax.set_yticks([0])  # Set a single y-tick at position 0
        ax.set_yticklabels([y_names[k]], rotation=90, va='center')

        if k == num_classes - 1:
            ax.set_xticks(np.arange(len(axis_names)))
            ax.set_xticklabels(axis_names, rotation=90)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

    # Position the colorbar on top
    cbar_ax = fig.add_axes([0.2, 1, 0.6, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, location='top')

    # Customize the legend labels
    vmin = min(vmin, heatmap.min())
    vmax = max(vmax, heatmap.max())
    cbar.set_ticks([vmin, vmax])  # Set the tick positions
    cbar.set_ticklabels(['Least Important', 'Most Important'], rotation=90)
    cbar.ax.invert_xaxis()

    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the spacing between subplots

    plt.savefig(file_name,bbox_inches='tight')

def plot_stacked_csv(stacked_array, num_classes, path,file_name):
    y_names, c = get_classes_labels_csv(path, num_classes)
    axis_names = c
    fig, axes = plt.subplots(num_classes, 1, figsize=(8, 6))
    vmin = float('inf')  # Initialize the minimum value
    vmax = float('-inf')  # Initialize the maximum value

    for k in range(num_classes):
        heatmap = stacked_array[k][0].mean(axis=0)
        ax = axes[k]
        im = ax.imshow(heatmap, cmap='Blues')
        ax.set_yticks([0])  # Set a single y-tick at position 0
        ax.set_yticklabels([y_names[k]], rotation=90, va='center')

        if k == num_classes - 1:
            ax.set_xticks(np.arange(len(axis_names)))
            ax.set_xticklabels(axis_names, rotation=90)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

    # Position the colorbar on top
    cbar_ax = fig.add_axes([0.2, 1, 0.6, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, location='top')

    # Customize the legend labels
    vmin = min(vmin, heatmap.min())
    vmax = max(vmax, heatmap.max())
    cbar.set_ticks([vmin, vmax])  # Set the tick positions
    cbar.set_ticklabels(['Least Important', 'Most Important'], rotation=90)
    cbar.ax.invert_xaxis()

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the spacing between subplots

    plt.savefig(file_name,bbox_inches='tight')

def weighted_features_sum(W,P,num_classes):
    features = [[] for i in range(num_classes)]

    f=0

    for i in range(len(P)):
        for j in range (len(W[0][i])):

        #if torch.argmax(P[i][0][j]).item()== torch.argmax(P[i][1][j]).item():
            #print("Correct prediction, Model vis : ",torch.argmax(P[i][0][j]).item()+1 )
            sum_w=0
            for k in range(num_classes):
                sum_w+=W[k][i].detach().numpy()[j]
            weighted = np.dot(sum_w,np.ones(len(W[0][0][0])))
            weighted = weighted.reshape(-1, 1)
            scaler = StandardScaler()
            standardized_array = scaler.fit_transform(weighted)
            data = np.array(standardized_array).reshape(1, len(standardized_array))
            features[torch.argmax(P[i][1][j]).item()].append(data)
        #else:
            #f+=1

    return features

def stack_all_folds(folder,num_classes,K):
    all_folds= [[] for i in range(K)]
    stacked_array= [[] for i in range(num_classes)]
    for fold in range(K):
        W=torch.load(r"\runs_episodes\\" +folder +"\\Weights_"+str(fold+1)+".pt")
        P=torch.load(r"\runs_episodes\\" +folder +"\\P_Labels_"+str(fold+1)+".pt")
        all_folds[fold].append(weighted_features_sum(W,P,num_classes))
    for j in range(num_classes):
        stacked_array[j].append(np.concatenate([all_folds[i][0][j] for i in range(K)]))
    return stacked_array

def Plot_Attention_weights(folder,folds,task,is_openml,file_name,num_classes=0):
    if is_openml == True:
        task_id,num_classes = get_dataset_id_multi(int(task))
        stacked_array = stack_all_folds(folder,num_classes,folds)
        plot_stacked(stacked_array, num_classes, int(task),file_name)
    else:
        stacked_array = stack_all_folds(folder,num_classes,folds)
        plot_stacked_csv(stacked_array,num_classes,task,file_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', type=str,
        help='attributes to  plot')
    parser.add_argument('-f', '--folder', type=str,
        help='folder location')
    parser.add_argument('-t', '--task',
        help='task number or csv file path')
    parser.add_argument('-nc', '--num_classes', type=int, default=0,
        help='attributes to  plot')
    parser.add_argument('-fo', '--folds', type=int, default=10,
        help='number of folds')
    parser.add_argument('-sn', '--save_name', type=str,
        help='folder location')
    parser.add_argument('-io', '--is_openml',type=bool,
        help='True if data from openml')

    args = vars(parser.parse_args())

    if args['plot']== 'auc':
        #plot auc
        print(args['task'])
        Plot_AUC(folder=args['folder'],task=args['task'],is_openml=args['is_openml'],file_name=args['save_name'])
    elif args['plot']== 'att':
        #plot attention weights
        print(args['task'])
        Plot_Attention_weights(folder=args['folder'],folds=args['folds'],task=args['task'],is_openml=args['is_openml'],file_name=args['save_name'],num_classes=args['num_classes'])
