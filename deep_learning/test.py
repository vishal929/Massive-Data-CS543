# test.py holds code to evaluate our neural networks on a test set

from tqdm import tqdm
from helper import load_model
import torch
from torch.utils.data import DataLoader
from process_input import Airplane_Weather_Dataset

# we provide the model_name, data_loader for test data
# the task is specified in the model dump so no need to specify it as a parameter

# for categorical, we can output loss and accuracy
# for regression we just output the MSE (mean squared error)
def evaluate(model,data_loader,task,device):
    # setting the model to evaluate mode
    model.eval()

    loss = 0

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

# we are interested to see if the regression model can get the outputs correct for any categorical set
# basically we are using a regression model (which should be more powerful) for the categorical task
def special_evaluate_regression():
    # setting the model to evaluate mode
    print('we are running special categorical testing of a regression model!')
    model.eval()

    # we also would like to see accuracy here
    num_total = 0
    num_correct = 0
    with torch.no_grad():
        with tqdm(data_loader, unit='batch') as data:
            for record, target in data:
                record = record.to(device)
                target = target.to(device)
                res = model(record)

                # we have observations of (batch_size,1) so lets just squeeze this
                res = torch.squeeze(res)

                # we want to see if the regression model offers powerful categorical predictions!
                num_total += record.shape[0]
                # we have false if the output is zero or less and true if greater
                guesses = res > 0
                actual = target > 0
                num_correct += torch.sum((guesses == actual).long())

    cat_accuracy = num_correct / num_total
    print('regression model as categorical predictor accuracy: ' + str(cat_accuracy))
    # setting model back to training mode (in case this is used for something else)
    model.train()

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

# calling evaluate
evaluate(model,data_loader,task,device)
