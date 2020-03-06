''' CS 535 Final Project: Storm Surge Prediction Template File
Created By: Dylan Sanderson, Derek Jackson, Meredith Leung
'''

from __future__ import print_function
from __future__ import division
import os
import pandas as pd
import numpy as np
import scipy.io as io
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
# from tensorboard import SummaryWriter  # for pytorch below 1.14
from torch.utils.tensorboard import SummaryWriter # for pytorch above or equal 1.14
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from data_reader import CHS_DataSet

class Net(nn.Module):
    '''
    Net Summary:
    Net Input: CentralPressureDeficit, RadiusMaxWinds, TranslationalSpeed
    Net Output: Max Surge values
    '''

    def __init__(self, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 200)
        self.fc3 = nn.Linear(200, 512)
        self.fc4 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# evaluates regression type prediction
def eval_net(dataloader):
    correct = 0
    total = 0
    total_loss = 0
    net.eval() # Why would I do this? To switch model to eval mode
    criterion = nn.MSELoss()
    for data in dataloader:
        storm_data, labels = data.float()
        storm_data, labels = Variable(storm_data).cuda(), Variable(labels).cuda()
        outputs = net(storm_data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    net.train() # Why would I do this? To switch model back to train mode
    return total_loss / total, correct.float() / total

if __name__ == "__main__":
    BATCH_SIZE = 50 #mini_batch size
    MAX_EPOCH = 10 #maximum epoch to train

    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
    #                                           shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
    #                                          shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # path to data
	# path_to_data = os.path.join(os.getcwd(), '..', 'data')
    path_to_data = os.path.join('E:\Documents\Box Sync\CS535_Project', 'data')
	
    """ defining bounding box """
    # # large bounding box
    # xmin, xmax = -74.619, -73.397
    # ymin, ymax = 40.080, 40.892

    # smaller bounding box
    xmin, xmax = -74.2754, -73.9374
    ymin, ymax = 40.4041, 40.6097

    train_test_split = 0.8		# ratio to split test and train data

    # dataset class
    dataset = CHS_DataSet(path_to_data, xmin, xmax, ymin, ymax, max_surge=True)
    output_size = len(dataset.target[0]) #output size, needed to configure model
    print('Size of Output: {}'.format(output_size))
    print('setup dataset class')

    # computing size of train and test datasets
    train_size = int(train_test_split * len(dataset))
    test_size = len(dataset) - train_size
    lengths = [train_size, test_size]

    # splitting the data into train and test sets
    trn_ds, tst_ds = random_split(dataset, lengths)
    print('split data into train and test sets')

    # setting up train and test dataloaders
    trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    tst_loader = DataLoader(tst_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)



    print('Building model...')
    net = Net(output_size).cuda()
    net = net.float()
    net.train() # Why would I do this? To make sure model is in train mode

    writer = SummaryWriter(log_dir='./log/template')
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trn_loader, 0):
            # get the inputs
            inputs, targets = data

            # wrap them in Variable
            inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())
            loss = criterion(outputs.float(), targets.float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                        (i + 1, running_loss / 500))
                running_loss = 0.0

        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trn_loader)
        test_loss, test_acc = eval_net(tst_loader)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
                (epoch+1, train_loss, train_acc, test_loss, test_acc))

        writer.add_scalars('Loss', {'Train':train_loss ,'Test':test_loss}, epoch+1)
        writer.add_scalars('Accuracy', {'Train':train_acc ,'Test':test_acc}, epoch+1)

    writer.close()
    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), 'template_model.pth')
