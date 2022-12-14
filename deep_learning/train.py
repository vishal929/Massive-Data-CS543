# train.py holds code to train our neural networks

# we use tqdm to provide a progress bar type interface
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from process_input import Airplane_Weather_Dataset
from helper import dump_model, load_model
import pandas as pd

# saving train losses and validation losses to text file (for later graphing)
def save_losses(train_losses,val_losses):
    df = pd.DataFrame(list(zip(train_losses,val_losses)),columns=['train_loss','val_loss'])
    df.to_csv('./training_history.csv')

# function to get validation loss
def get_validation_loss(model,data_loader,task,device):
    val_loss = 0
    # setting the model to evaluate mode (we do not want dropout when evaluating)
    model.eval()
    with torch.no_grad():
        with tqdm(data_loader, unit='batch') as data:
            for data_record, label in data:
                data_record = data_record.to(device)
                target = label.to(device)
                # we have our training data, running the model
                res = model(data_record)

                # we have (batch,output=1) so lets just squeeze this to (batch)
                res = torch.squeeze(res)

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

                del data_record
                del target
    print('got val_loss: ' + str(val_loss))

    # returning the model to training mode
    model.train()
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
    memory_usage_printed = False
    val_loss = None
    # basically if there are 5 consecutive epochs where validation loss is worse, we stop
    early_stopping_patience = 5
    early_stopping_count = 0
    # keeping track of train losses and val losses for every epoch
    train_losses = []
    val_losses = []
    for i in tqdm(range(num_epochs_completed,num_epochs_total)):
        train_loss = 0
        with tqdm(data_loader,unit='batch') as data:
            for data_record,label in data:
                data_record = data_record.to(device)
                target = label.to(device)
                if not memory_usage_printed:
                    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
                    print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
                    print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
                    memory_usage_printed = True
                # we have our training data, running the model
                res = model(data_record)

                # we have (batch,output) so lets just squeeze this
                res = torch.squeeze(res)

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

                # removing the batch from memory
                del data_record
                del target
        num_epochs_completed += 1
        print('train_loss on epoch:' + str(num_epochs_completed)+', loss: ' + str(train_loss))
        # getting validation loss (if this is worse than our last validation loss we stop)
        new_val_loss = get_validation_loss(model,val_data_loader,task,device)

        train_losses.append(train_loss.item())
        val_losses.append(new_val_loss.item())

        if val_loss is not None and new_val_loss > val_loss:
            early_stopping_count += 1
            if early_stopping_count == early_stopping_patience:
                # our model is doing worse on validation data, we will stop here!
                print('validation loss was worse than last one for our patience! We will stop training here!')
                # saving our train losses and val losses to some txt file
                save_losses(train_losses,val_losses)
                return
        early_stopping_count = 0

        # updating validation loss for next epoch
        val_loss = new_val_loss
        print('validation loss on epoch:' + str(num_epochs_completed) + ', loss: ' + str(val_loss))

        # after a certain # of epochs we save our model to disk
        if num_epochs_completed % num_epoch_save_interval == 0:
            # then we dump the model to disk
            dump_model(model,num_epochs_completed,optimizer,learning_rate,task,model_name)

        if num_epochs_completed == num_epochs_total:
            print('we hit the epoch limit... we will stop training!')
            # saving losses to csv
            save_losses(train_losses,val_losses)


# creating data loader for both train and validation
train_set = Airplane_Weather_Dataset('regression','train')
validation_set = Airplane_Weather_Dataset('regression','validation')

print('obtained our training and validation sets!')

model_name = 'test_regression_3'
task = 'regression'
learning_rate = 0.02
num_hidden = 5
num_hidden_features = 300
input_features = 15
batch_size = 524288

num_epochs_total = 600
num_epoch_save_interval = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model,optimizer, learning_rate, num_epochs_completed, task= \
    load_model(model_name,device,task,learning_rate,num_hidden,num_hidden_features,input_features)

print('loaded our model and optimizer!')
print('our model has : ' + str(num_epochs_completed) + ' num epochs already completed!')

# sending parameters etc. to device

# creating dataloaders
# we should shuffle our train data to vary from epoch to epoch
data_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=8)

# we dont need to shuffle validation data since its only used for evaluation
val_data_loader = DataLoader(validation_set,batch_size=batch_size, num_workers=8)

print('created our data loaders! Lets start training...')

# now we can call train
train(model,data_loader,val_data_loader,num_epochs_completed,num_epochs_total,num_epoch_save_interval,
      task,optimizer,model_name,device)