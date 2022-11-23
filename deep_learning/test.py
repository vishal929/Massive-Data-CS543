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
    if task == 'categorical':
        # we also would like to see accuracy here
        num_total = 0
        num_correct = 0
    else:
        num_total = None
        num_correct = None
    with torch.no_grad():
        with tqdm(data_loader,unit='batch') as data:
            for record,target in data:
                record = record.to(device)
                target = target.to(device)
                res = model(record)

                # we have observations of (batch_size,1) so lets just squeeze this
                res = torch.squeeze(res)

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
    # setting model back to training mode (in case this is used for something else)
    model.train()


# getting data loader for test data
batch_size = 4194304
task = 'categorical'
dataset = Airplane_Weather_Dataset(task,'test')
data_loader = DataLoader(dataset,batch_size=batch_size,num_workers=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading model from disk
model_name = 'test_categorical'
model, _, lr, num_epochs_completed, task = load_model(model_name,device)

print('loaded model: ' + str(model_name) + ' with learning rate: ' + str(lr) + \
      ' with num epochs completed: ' + str(num_epochs_completed))

# calling evaluate
evaluate(model,data_loader,task,device)
