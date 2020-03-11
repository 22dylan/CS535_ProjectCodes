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

torch.manual_seed(1337)
class Net(nn.Module):
    """
    Net Summary:
    Net Input: 337x9
        - each row correponds to time step.
        - 9 input columns correspond to:
            0) Central Pressure
            1) Far Field pressure
            2) Forward Speed
            3) Heading
            4) Holland B1
            5) Radius Max Winds
            6) Radius pressure diff
            7) Latitude
            8) Longitude
    Net Output: Max Surge values at each save point
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=n_layers, 
                          batch_first=True
                        )

        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size = np.shape(x)[0]
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                  torch.zeros(self.n_layers, batch_size, self.hidden_size))
        x = x.float()
        out, hidden = self.lstm(x, hidden)
        out = out[:,-1,:]
        out = self.fc(out)
        return out

# evaluates regression type prediction
def eval_net(dataloader, gpu):
    correct = 0
    total = 0
    avg_loss = 0
    net.eval()
    criterion = nn.MSELoss(reduction='mean')
    for data in dataloader:
        inputs, targets = data
        if gpu==True:
            inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = net(inputs.float())
        predicted = outputs[:]
        # _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        # correct += (predicted == targets.data).sum()
        correct += abs(targets - predicted).sum()

        loss = criterion(outputs.float(), targets.float())
        avg_loss += loss.item()
    net.train() # Why would I do this? To switch model back to train mode

    # correct = 0 #should delete when can
    # total = 1 #should delete when can
    # return avg_loss, correct.float() / total
    return avg_loss/total, correct/total, outputs, targets

if __name__ == "__main__":
    gpu = False
    BATCH_SIZE = 50     # mini_batch size
    MAX_EPOCH = 100      # maximum epoch to train
    hidden_size = 25    # size of hidden layer
    n_layers = 1        # number of lstm layers

    key = 'B{}_h{}_l{}' .format(BATCH_SIZE, hidden_size, n_layers)

    # path to data
    path_to_data = os.path.join(os.getcwd(), '..', 'data')
    # path_to_data = os.path.join(os.getcwd(), 'data')
	
    """ defining bounding box """
    # # large bounding box
    # xmin, xmax = -74.619, -73.397
    # ymin, ymax = 40.080, 40.892

    # smaller bounding box
    xmin, xmax = -74.2754, -73.9374
    ymin, ymax = 40.4041, 40.6097

    train_test_split = 0.8		# ratio to split test and train data

    # dataset class
    dataset = CHS_DataSet(path_to_data, 
                            xmin, xmax, 
                            ymin, ymax, 
                            ts_input=True,
                            ts_output=False, 
                            pad_type=0.0)
    
    input_size = np.shape(dataset.storm_conds)[2]  # number of input
    output_size = len(dataset.target[0]) #output size, needed to configure model
    print('Number of input dimensions at each time step: {}' .format(input_size))
    print('Size of Output: {} save points'.format(output_size))

    # computing size of train and test datasets
    train_size = int(train_test_split * len(dataset))
    test_size = len(dataset) - train_size
    lengths = [train_size, test_size]
    print('Training examples: {}  Testing examples: {}'.format(train_size, test_size))

    # splitting the data into train and test sets
    trn_ds, tst_ds = random_split(dataset, lengths)

    # setting up train and test dataloaders
    trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    tst_loader = DataLoader(tst_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print('Building model...')
    print('\tinput_size: {}' .format(input_size))
    print('\thidden_size: {}' .format(hidden_size))
    print('\toutput_size/num save points: {}' .format(output_size))
    print('\tn_layers: {}' .format(n_layers))

    if gpu == True:
        net = Net(input_size=input_size, 
                  hidden_size = hidden_size,
                  output_size=output_size, 
                  n_layers=n_layers).cuda()

    net = Net(input_size=input_size, 
              hidden_size = hidden_size,
              output_size=output_size, 
              n_layers=n_layers)

    net = net.float()
    net.train()

    writer = SummaryWriter(log_dir='./log/template')
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    epoch_out = []
    test_acc_out = []
    train_acc_out = []
    test_loss_out = []
    train_loss_out = []

    print('Start training...')
    iii = 0 # counter for tensorboard plotting
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trn_loader, 0):
            # get the inputs
            inputs, targets = data

            # wrap them in Variable
            if gpu == True:
                inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs.float(), targets.float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 100 == 0:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                        (i + 1, running_loss / 500))
                running_loss = 0.0

        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc, outputs, targets = eval_net(trn_loader, gpu)
        test_loss, test_acc, outputs, targets = eval_net(tst_loader, gpu)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
                (epoch+1, train_loss, train_acc, test_loss, test_acc))

        epoch_out.append(epoch+1)
        test_acc_out.append(float(test_acc))
        train_acc_out.append(float(train_acc))
        test_loss_out.append(test_loss)
        train_loss_out.append(train_loss)

        # writer.add_scalars('Loss', {'Train':train_loss ,'Test':test_loss}, epoch+1)
        # writer.add_scalars('Accuracy', {'Train':train_acc ,'Test':test_acc}, epoch+1)
        # print('outputs size: {}'.format(outputs.size(0)))
        # print('outputs size: {}'.format(surge_levels.size(0)))
        # for ii in range(outputs.size(0)):
        #     writer.add_scalars('Comparing Predictions', {'Prediction': outputs[ii][0], 'Reality': surge_levels[ii][0]},iii)
        #     iii+=1

    output = pd.DataFrame()
    output['epoch'] = epoch_out
    output['test_acc'] = test_acc_out
    output['train_acc'] = train_acc_out
    output['test_loss'] = test_loss_out
    output['train_loss'] = train_loss_out
    path_out = os.path.join('LSTM_training_results', '{}_results.csv' .format(key))
    output.to_csv(path_out, index=False)

    writer.close()
    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), 'template_model.pth')
