# test.py holds code to evaluate our neural networks on a test set

from tqdm import tqdm
from helper import load_model
import torch
import numpy as np
from torch.utils.data import DataLoader
from process_input import Airplane_Weather_Dataset
import pandas as pd

# we provide the model_name, data_loader for test data
# the task is specified in the model dump so no need to specify it as a parameter

# for categorical, we can output loss and accuracy
# for regression we just output the MSE (mean squared error)
def evaluate(model,data_loader,task,device):
    # setting the model to evaluate mode
    model.eval()

    loss = 0
    guesses = []
    actual = []
    # we also would like to see accuracy here
    num_total = 0
    num_correct = 0
    with torch.no_grad():
        with tqdm(data_loader,unit='batch') as data:
            for record,target in data:
                record = record.to(device)
                target = target.to(device)
                res = model(record)

                # we have observations of (batch_size,1) so lets just squeeze this
                res = torch.squeeze(res)



                if task == 'categorical':
                    loss += torch.nn.BCELoss()(res,target)
                    # adding the batch size to the num_total, because we make batch_size guesses
                    num_total += record.shape[0]
                    guesses = torch.round(res)
                    num_correct += torch.sum((target == guesses).long())

                elif task == 'regression':
                    loss += torch.nn.MSELoss()(res,target)
                    # we also want to see if the regression model offers powerful categorical predictions!
                    num_total += record.shape[0]
                    # we have false if the ouput is zero or less and true if greater
                    guesses = res > 0
                    actual = target > 0
                    num_correct += torch.sum((guesses == actual).long())

                guesses.extend(res.tolist())
                actual.extend(target.tolist())
    if task == 'categorical':
        accuracy = num_correct/num_total
        print('categorical loss: ' + str(loss))
        print('categorical accuracy: ' + str(accuracy))
    elif task == 'regression':
        print('regression loss: ' + str(loss))
        cat_accuracy = num_correct/num_total
        #print('regression model as categorical predictor accuracy: ' + str(cat_accuracy))
    # setting model back to training mode (in case this is used for something else)
    model.train()
    return guesses,actual

# for regression, we want to not only see loss,  but within 5-minute accuracy and within 10-minute accuracy
# in addition, we want to just observe the first five guesses and the first five actual targets (for reporting)
def get_attributes_for_regression(guesses,actual):
    data = pd.DataFrame(data=[guesses,actual],columns=['predicted','actual'])
    # we want to read in training statistics so that we can "denormalize" these
    train_dep_delay = pd.read_parquet(task + '_' + 'train')['DepDelay']
    train_max = train_dep_delay.max()
    train_min = train_dep_delay.min()

    #denormalization
    data = (data * (train_max-train_min)) + train_min

    # printing top 10 predictions and actual
    print(data.head(10))

    # getting within 5 minute accuracy
    data['5-min-bool'] = data.apply(lambda x: x['actual']-5 <= x['predicted'] <= x['actual']+5)
    print('model 5-min accuracy: ' + str(data['5-min-bool'].sum()/data.size()))
    # getting within 10 minute accuracy
    data['10-min-bool'] = data.apply(lambda x: x['actual'] - 10 <= x['predicted'] <= x['actual']+10)
    print('model 10-min accuracy: ' + str(data['10-min-bool'].sum() / data.size()))

# getting data loader for test data
batch_size = 524288
task = 'regression'
dataset = Airplane_Weather_Dataset(task,'test')
data_loader = DataLoader(dataset,batch_size=batch_size,num_workers=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading model from disk
model_name = 'test_regression_3'
model, _, lr, num_epochs_completed, task = load_model(model_name,device)

print('loaded model: ' + str(model_name) + ' with learning rate: ' + str(lr) + \
      ' with num epochs completed: ' + str(num_epochs_completed))

# we want to see the number of trainable parameters
model_parameters = filter(lambda parameter: parameter.requires_grad, model.parameters())
num_params = sum([np.prod(parameter.size()) for parameter in model_parameters])
print('the model has: ' + str(num_params) + ' number of trainable parameters!')


# calling evaluate
guesses,actual = evaluate(model,data_loader,task,device)
if task == 'regression':
    get_attributes_for_regression(guesses,actual)