# train.py holds code to train our neural networks

# we use tqdm to provide a progress bar type interface
from tqdm import tqdm
import torch.nn as nn
from helper import dump_model, load_model


# function to save a model to disk

# train function
# we input our model, dataloader, current epochs and number of epochs total we want to run it for
# task should be one of 'classification', or 'regression'
# we include an optimizer for actually performing the descent
# we provide a save interval for the model (i.e every 5 epochs save the model to disk)
# we pass a model_name so that we can dump it to some file and save progress
def train(model, data_loader, num_epochs_completed, num_epochs_total, num_epoch_save_interval, task, optimizer, model_name):

    # number of epochs
    for i in tqdm(range(num_epochs_completed,num_epochs_total)):
        with tqdm(data_loader,unit='batch') as data:
            for data_record,label in data:
                target = None
                # we have our training data, running the model
                res = model.forward(data_record)

                # zeroing grad before descent
                optimizer.zero_grad()

                # calculating our loss
                if task == 'classification':
                    # we want to use binary cross entropy loss here for classification
                    loss = nn.BCELoss(res,target)
                elif task == 'regression':
                    # we want to use mean squared error loss here for regression
                    loss = nn.MSELoss(res,target)
                else:
                    # bad case
                    loss = None

                # calling backward and step to update weights according to loss
                loss.backward()
                optimizer.step()

        num_epochs_completed += 1

        # after a certain # of epochs we save our model to disk
        if num_epochs_completed % num_epoch_save_interval == 0:
            # then we dump the model to disk
            dump_model(model,num_epochs_completed,optimizer,learning_rate,task,model_name)


# creating data loader
data_loader = None

model_name = 'test_categorical'
task = 'categorical'
learning_rate = 0.3
num_hidden = 3
num_hidden_features = 500
input_features = 15

num_epochs_total = 200
num_epoch_save_interval = 5

model,optimizer, learning_rate, num_epochs_completed, task = \
    load_model(model_name,task,learning_rate,num_hidden,num_hidden_features,input_features)

# now we can call train
train(model,data_loader,num_epochs_completed,num_epochs_total,num_epoch_save_interval,task,optimizer,model_name)