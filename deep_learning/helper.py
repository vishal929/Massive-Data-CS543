# helper function to load a model from disk into pytorch and to dump a model to disk
import torch
import os.path

from regression_model import Airline_Weather_Regression_Model
from categorical_model import Airline_Weather_Categorical_Model


# we save the task with the model (either 'classification' or 'regression')
# num_hidden is the number of hidden layers
# num_hidden_features is the number of features in each hidden layer
def dump_model(model, num_epochs_completed, optimizer, learning_rate, task, model_name):
    num_hidden_features = model.num_hidden_features
    num_hidden = model.num_hidden
    input_features = model.input_features
    torch.save({
        'num_epochs_completed': num_epochs_completed,
        'optimizer_state': optimizer.state_dict(),
        'learning_rate': learning_rate,
        'model_state': model.state_dict(),
        'num_hidden': num_hidden,
        'num_hidden_features': num_hidden_features,
        'input_features': input_features,
        'task': task
    }, model_name + '.pth')


# we load the model if the specified model is saved to disk
# otherwise, we create a new model with the specified task
def load_model(model_name, device, task=None, learning_rate=0.3,
               num_hidden=3, num_hidden_features=500, input_features=15):
    if os.path.exists(model_name + '.pth'):
        # then the model is dumped, we can load it
        state = torch.load(model_name + '.pth')
        print(state)
        num_epochs_completed = state['num_epochs_completed']
        num_hidden = state['num_hidden']
        num_hidden_features = state['num_hidden_features']
        input_features = state['input_features:']
        task = state['task']
        if task == 'categorical':
            # load classification model
            model = Airline_Weather_Categorical_Model(num_hidden, input_features, num_hidden_features)\
                .to(device)
            model.load_state_dict(state['model_state'])
        elif task == 'regression':
            # load the regression model
            model = Airline_Weather_Regression_Model(num_hidden, input_features, num_hidden_features)\
                .to(device)
            model.load_state_dict(state['model_state'])
        else:
            model = None

        # getting the optimizer state
        learning_rate = state['learning_rate']
        optimizer = torch.optim\
            .SGD(lr=learning_rate, params=model.parameters())
        optimizer.load_state_dict(state['optimizer_state'])

        # returning what we need to run training (or maybe evaluation)
        return model, optimizer, learning_rate, num_epochs_completed, task

    # if the dump does not exist, we create a model from scratch, an optimizer and return it
    if task == 'categorical':
        model = Airline_Weather_Categorical_Model(num_hidden,input_features,num_hidden_features)\
            .to(device)
    elif task == 'regression':
        model = Airline_Weather_Regression_Model(num_hidden,input_features,num_hidden_features)\
            .to(device)
    else:
        model = None

    optimizer = torch.optim.SGD(lr = learning_rate,params = model.parameters())
    return model,optimizer, learning_rate, 0, task
