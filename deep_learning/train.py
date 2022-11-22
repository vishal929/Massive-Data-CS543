# train.py holds code to train our neural networks

# we use tqdm to provide a progress bar type interface
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from process_input import Airplane_Weather_Dataset
from helper import dump_model, load_model

# function to get validation loss
def get_validation_loss(model,data_loader,task,device):
    val_loss = 0
    with torch.no_grad():
        with tqdm(data_loader, unit='batch') as data:
            for data_record, label in data:
                data_record = data_record.to(device)
                target = label.to(device)
                # we have our training data, running the model
                res = model(data_record)

                # calculating our loss
                if task == 'categorical':
                    # we want to use binary cross entropy loss here for classification
                    loss = torch.nn.BCELoss()(res,target)
                elif task == 'regression':
                    # we want to use mean squared error loss here for regression
                    loss = torch.nn.MSELoss()(res,target)
                else:
                    # bad case
                    loss = None
                val_loss += loss
    # returning the loss
    return val_loss


# train function
# we input our model, dataloader, current epochs and number of epochs total we want to run it for
# task should be one of 'classification', or 'regression'
# we include an optimizer for actually performing the descent
# we provide a save interval for the model (i.e every 5 epochs save the model to disk)
# we pass a model_name so that we can dump it to some file and save progress
def train(model, data_loader, val_data_loader, num_epochs_completed, num_epochs_total, num_epoch_save_interval,
          task, optimizer, model_name, device):

    # number of epochs
    for i in tqdm(range(num_epochs_completed,num_epochs_total)):
        train_loss = 0
        val_loss = None
        with tqdm(data_loader,unit='batch') as data:
            for data_record,label in data:
                data_record = data_record.to(device)
                target = label.to(device)
                # we have our training data, running the model
                res = model(data_record)

                # zeroing grad before descent
                optimizer.zero_grad()

                # calculating our loss
                if task == 'categorical':
                    # we want to use binary cross entropy loss here for classification
                    loss = torch.nn.BCELoss()(res,target)
                elif task == 'regression':
                    # we want to use mean squared error loss here for regression
                    loss = torch.nn.MSELoss()(res,target)
                else:
                    # bad case
                    loss = None
                train_loss += loss
                # calling backward and step to update weights according to loss
                loss.backward()
                optimizer.step()
        num_epochs_completed += 1
        print('train_loss on epoch:' + str(num_epochs_completed)+', loss: ' + str(train_loss))
        # getting validation loss (if this is worse than our last validation loss we stop)
        new_val_loss = get_validation_loss(model,val_data_loader,task,device)
        print('validation loss on epoch:' + str(num_epochs_completed) + ', loss: ' + str(val_loss))
        if val_loss is not None and new_val_loss > val_loss:
            # our model is doing worse on validation data, we will stop here!
            print('validation loss was worse than last one! We will stop training here!')
            return

        # updating validation loss for next epoch
        val_loss = new_val_loss

        # after a certain # of epochs we save our model to disk
        if num_epochs_completed % num_epoch_save_interval == 0:
            # then we dump the model to disk
            dump_model(model,num_epochs_completed,optimizer,learning_rate,task,model_name)


# creating data loader for both train and validation
train_set = Airplane_Weather_Dataset('categorical','train')
validation_set = Airplane_Weather_Dataset('categorical','validation')

print('obtained our training and validation sets!')

model_name = 'test_categorical'
task = 'categorical'
learning_rate = 0.3
num_hidden = 3
num_hidden_features = 30
input_features = 15
batch_size = 512

num_epochs_total = 100
num_epoch_save_interval = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model,optimizer, learning_rate, num_epochs_completed, task= \
    load_model(model_name,device,task,learning_rate,num_hidden,num_hidden_features,input_features)

print('loaded our model and optimizer!')

# sending parameters etc. to device

# creating dataloaders
# we should shuffle our train data to vary from epoch to epoch
data_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=4)

# we dont need to shuffle validation data since its only used for evaluation
val_data_loader = DataLoader(validation_set,batch_size=batch_size, num_workers=4)

print('created our data loaders! Lets start training...')

# now we can call train
train(model,data_loader,val_data_loader,num_epochs_completed,num_epochs_total,num_epoch_save_interval,
      task,optimizer,model_name,device)