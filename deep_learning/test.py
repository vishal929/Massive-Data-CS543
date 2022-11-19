# test.py holds code to evaluate our neural networks on a test set

from tqdm import tqdm
from helper import load_model


# we provide the model_name, data_loader for test data
# the task is specified in the model dump so no need to specify it as a parameter

# for categorical, we can output loss and accuracy
# for regression we just output the MSE (mean squared error)
def evaluate(model,data_loader,task):
    with tqdm(data_loader,unit='batch') as data:
        for record,target in data:
            res = model(record)

            if task == 'categorical':
                pass
            elif task == 'regression':
                pass

# getting data loader for test data
data_loader = None
# loading model from disk

# calling evaluate
