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
def evaluate(model,data_loader,task):
    loss = 0
    if task == 'categorical':
        # we also would like to see accuracy here
        num_total = 0
        num_correct = 0
    with torch.no_grad():
        with tqdm(data_loader,unit='batch') as data:
            for record,target in data:
                res = model(record)

                if task == 'categorical':
                    loss += torch.nn.BCELoss(res,target)
                    # adding the batch size to the num_total, because we make batch_size guesses
                    num_total += record.shape[0]
                    guesses = torch.round(res)
                    num_correct += torch.sum(target == guesses)

                elif task == 'regression':
                    loss += torch.nn.MSELoss(res,target)
    if task == 'categorical':
        accuracy = num_correct/num_total
        print('categorical loss: ' + str(loss))
        print('categorical accuracy: ' + str(accuracy))
    elif task == 'regression':
        print('regression loss: ' + str(loss))


# getting data loader for test data
task = 'categorical'
dataset = Airplane_Weather_Dataset(task,'test')
data_loader = DataLoader(dataset,batch_size=100)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading model from disk
model_name = 'test_categorical'
model = load_model(model_name=model_name)


# calling evaluate
evaluate(model,data_loader,task)
