import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split
import torchvision.transforms as tr
import torchvision.transforms.functional as fc
from torchvision.io import read_image
from torchvision.datasets import ImageFolder

from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

from time import time
from random import random


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# test data just transfromed into the same size 
test_transform = tr.Compose([
    tr.Resize(size=(96,96), antialias=True),
    tr.Normalize(mean=[0.299], std=[0.137])
    ])
# transforms for data augmentation
augm_transform = (
    tr.Compose([
    tr.Resize(size=(99, 99), antialias=True),
    tr.RandomVerticalFlip(p=0.8),
    tr.RandomCrop(size=(96,96)),
    tr.Normalize(mean=[0.299], std=[0.137])
]),
tr.Compose([
    tr.Resize(size=(99, 99), antialias=True),
    tr.RandomHorizontalFlip(p=0.8),
    tr.RandomCrop(size=(96,96)),
    tr.Normalize(mean=[0.299], std=[0.137])
]),
tr.Compose([
    tr.Resize(size=(112,112), antialias=True),
    tr.RandomRotation(degrees=(3,9),expand=True),
    tr.CenterCrop(size=(96,96)),
    tr.Normalize(mean=[0.299], std=[0.137]),
    ]),
tr.Compose([
    tr.Resize(size=(112,112), antialias=True),
    tr.RandomPerspective(distortion_scale=0.15, p=0.8),
    tr.CenterCrop(size=(96,96)),
    tr.Normalize(mean=[0.299], std=[0.137]),
    ])
    )


def stan_image_loader(img_dir ='./data/STAN_patches_lbls/', train_bs = 16, split_p =0.8, data_3D = False):

    """ utility function for data curation,
    reads the images and labels from folders and creates test and train datasets and data loaders
    train data are augmented because of small size of current dataset
    Args:   img_dir | String - the relative path to main folder with data library
            train_bs  - train batch size; for the test dataset, all data will be loaded at once
            split | float - probability of inclusion of image in the training dataset
            data_3D | bool- if False, the data will be converted to the 2D 96x96 images of 5 channels(for loading into 2D CNNs),
                            if True,  the data will be converted to the 3D 5x96x96 images of 1 channel (for 3D CNN)

    Out: test_data_loader, train_data_loader (as Torch DataLoaders)
    """
    label_path = img_dir + 'labels/STAN_labels.csv'
    labels_df = pd.read_csv(label_path, delimiter=',', header=0)
    test_labels = []
    train_labels = []
    # 3D / 2D
    if data_3D:
        test_img_tensor = torch.empty((0,1,5,96,96), dtype=torch.float, device=device)
    else:
        test_img_tensor = torch.empty((0,5,96,96), dtype=torch.float, device=device)
    # 3D /2D
    if data_3D:
        train_img_tensor = torch.empty((0,1,5,96,96), dtype=torch.float, device=device)
    else:
        train_img_tensor = torch.empty((0,5,96,96), dtype=torch.float, device=device)
    # iterating through the folders and reading stack of monochrome images
    for idx, row in labels_df.iterrows():
        subfolder = labels_df.UID.iloc[idx]
        y_label = float(row['MUT_STATUS'])
        # the list of tensors representing a stack of 5 images(slices)
        imgs = []
        for sl in range(0,5):
            img_path = img_dir + 'patches/' + subfolder + f'/{subfolder}__sl_{sl}.png'
            img = fc.convert_image_dtype(read_image(img_path))
            imgs.append(img)
        # each list image is transformed into  [5, 96, 96] tensor to represent a single X feature
        load_img = torch.cat(imgs,dim=0)
        test_img = test_transform(load_img).unsqueeze_(0)

        # concatenating the features into a single test data tensor
        # 3D /2D
        if data_3D:
            test_img_tensor = torch.cat((test_img_tensor, torch.unsqueeze(test_img, 0)), dim=0)
        else:
            test_img_tensor = torch.cat((test_img_tensor, test_img), dim=0)
        # corresponding labels
        test_labels.append(y_label)

        # splitting 
        if random() <= split_p :
            #image included in the training set
            # for training data augmentation, each loaded 5-stack image is cloned and transformed into 4 different tensors
            for aug_tr in augm_transform:
                train_img = aug_tr(load_img).unsqueeze_(0)
                # concatenating the features into a single training data tensor
                # 3D / 2D
                if data_3D:
                    train_img_tensor = torch.cat((train_img_tensor, torch.unsqueeze(train_img,0)), dim=0)
                else:
                    train_img_tensor = torch.cat((train_img_tensor, train_img), dim=0)
                # corresponding labels
                train_labels.append(y_label)
    test_label_tensor = torch.Tensor(test_labels, device=device)
    train_label_tensor = torch.Tensor(train_labels, device=device)

    # data loaders to be used in the CNNs
    test_dataset = TensorDataset(test_img_tensor, test_label_tensor)
    train_dataset = TensorDataset(train_img_tensor, train_label_tensor)
    train_data_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size=train_bs, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size=len(test_labels), shuffle=True)
    
    return train_data_loader, test_data_loader


def kagg_brain_image_loader(img_dir = './data/brain_tumors/', train_bs = 64, split = 0.8):
    """ utility function for data curation,
    reads the images from Kaggle brain tumors data set and creates labels (0 - normal images)
    Args:   img_dir | String - the relative path to main folder with data library
            train_bs | int  - train batch size; for the test dataset, all data will be loaded at once
            split | float (0:1) - split factor
    Out: test_data_loader, train_data_loader (as Torch DataLoaders)
    """
    read_transform = tr.Compose([tr.ToTensor()])
    read_dataset = ImageFolder(img_dir, transform=read_transform)
    train_dataset, test_dataset = random_split(read_dataset, (split, 1-split))
    print(len(train_dataset), len(test_dataset))
    train_data_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size=train_bs, shuffle=False)
    test_data_loader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size=len(test_dataset), shuffle=True)
    return train_data_loader, test_data_loader

""" model evaluation utilities """


# accuracy evaluation
def accuracy_rate(predicted, true_y):
    """ takes predicted labels as torch model output (float) and
        converts them into integer category labels (typical for most of datasets) of shape [n_true_lables] 
        to conform the true lables
        Returns: accuracy score according to sklearn.metrics and converted predicted labels
    """
    if len(predicted.shape) > 1:
        pred_y = [np.argmax(p.detach().numpy()) for p in predicted]
    else:
        pred_y = [1 if pred > 0.5 else 0 for pred in predicted]
    acc_score = accuracy_score(true_y, pred_y)
    return acc_score, pred_y

# false negatives evaluation - important for the specific task (tumor classification)
def false_neg_rate(predicted, true_y):
    """ calculates the fraction of false negative labels on presumption that 
        label for normal images is 0
    """
    if len(predicted.shape) > 1:
        pred_y = [np.argmax(p.detach().numpy()) for p in predicted]
    else:
        pred_y = [1 if pred > 0.5 else 0 for pred in predicted]
    false_neg = [pr for pr, tr in zip(pred_y, true_y) if int(tr)>=1 and pr==0 ]
    return len(false_neg)/len(pred_y)

# showing some examples of classifications: first row - accurate, second - errorneous 
def show_examples(pred_labels, true_labels, test_images, show_cols=8, fig_prefix = 'brain_tumors', fig_idx=1 ):
    fig, axs = plt.subplots(2, show_cols, squeeze=False, figsize=(show_cols*2, 5) )
    true_count = 0
    err_count = 0
    idx = 0
    while true_count < show_cols or err_count < show_cols and idx < len(true_labels):
        tr_image = test_images[idx, :]
        # np_image = tr_image.permute(1,2,0)
        np_image = fc.to_pil_image(tr_image)
        # label =np.argmax(test_output[idx].detach().numpy())

        if pred_labels[idx] == true_labels[idx] and true_count < show_cols:
            axs[0,true_count].imshow(np_image)
            axs[0,true_count].set_title(f'label: {true_labels[idx]}')
            axs[0,true_count].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            true_count += 1
        if pred_labels[idx] != true_labels[idx] and err_count < show_cols:
            axs[1,err_count].imshow(np_image)
            axs[1,err_count].set_title(f'label: {true_labels[idx]}')
            axs[1,err_count].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            err_count += 1
        idx += 1
    plt.savefig(f'./reports/{fig_prefix}_examples_{fig_idx}.png', format='png')
    plt.savefig(f'./reports/{fig_prefix}_examples_{fig_idx}.svg', format='svg')
    plt.show()


def cnn_image_classifier(model, train_loader, test_loader, model_run_tag = '',
                     epochs=10, learn_rate= 0.01, criterion = nn.BCELoss()):
    
    """ The 'wrapper' function which performs model training and evaluation, 
            reports performance metrics, plots the learning curves and saves the model state
    Args:   
            CNN model, 
            train_loader - multiple batches of training data, 
            test_loader - single batch of data,
            model_run_tag - to be used in the file names for learning plots and model state
            number of epochs to train, learning rate, criterion 
    Returns: final accuracy, false negatives rate and total execution time
    """

    test_data = iter(test_loader)
    test_images, test_labels = next(test_data)
    # optimizer  = torch.optim.Adam(model.parameters(), lr=learn_rate)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learn_rate)
    start_tm = time()
    train_loss = []
    train_accuracy = []
    validation_accuracy = []
    for ep in range(epochs):  
        running_loss = 0
        ave_accuracy = 0  #epoch accuracy, averaged over batches
        with torch.no_grad():
            test_output = model(test_images.to(device))
        validation_accuracy.append(accuracy_rate(test_output, test_labels))
        for count, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            tr_outputs = model(images)
            acc_r, _ = accuracy_rate(tr_outputs, labels)
            ave_accuracy += acc_r
            loss = criterion(tr_outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss +=  loss.item()
        train_loss.append(running_loss/(count+1))
        train_accuracy.append(ave_accuracy/(count+1))
        print(f'loss: {running_loss} accuracy: {validation_accuracy[-1]}')
    end_tm = time()

    # reporting results
    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs[0].plot(range(1,epochs+1),train_loss)
    axs[0].set_title('loss over training cycle')

    axs[1].plot(range(1,epochs+1), train_accuracy, label="train accuracy")
    axs[1].plot(range(1,epochs+1), validation_accuracy, '--', label="test accuracy")
    axs[1].set_title('accuracy tracking')
    axs[1].legend()
    fig.suptitle(f'training of {model.name} model {model_run_tag}')
    plt.savefig(f'./reports/{model.name}_{model_run_tag}_ep{epochs}_lr_{int(1000*learn_rate)}.svg', format='svg')
    plt.show()
    torch.save(model.state_dict(), f'./models/{model.name}_{model_run_tag}.pkl')
    # evaluating model accuracy
    with torch.no_grad():
        test_output = model(test_images)
    
    exec_time = end_tm - start_tm
    final_accur, _ = accuracy_rate(test_output, test_labels)
    false_negs = false_neg_rate(test_output, test_labels)

    return final_accur, false_negs, exec_time

