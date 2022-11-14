# categorical model to predict if a delay will occur or not based on a record

# loss function for this should just be binary cross entropy


import torch
from standard_modules import FeedForward

# num_hidden is the number of hidden layers we want
# input_features is the number of features in the input
# hidden_features is the number of features we want for each hidden layer
class Airline_Weather_Categorical_Model(torch.nn.Module):
    def __init__(self,num_hidden,input_features,hidden_features):
        super(Airline_Weather_Categorical_Model, self).__init__()
        # creating the layers we want the model to have
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(num_hidden):
            if i==0:
                # initial feedforward
                self.hidden_layers.append(FeedForward(input_features,hidden_features))
            else:
                # rest of layers
                self.hidden_layers.append(FeedForward(hidden_features,hidden_features))

        # for categorical output we want to use a linear layer to reduce to dimension 1 followed by a sigmoid
        # basically if >0.5 we will say a delay exists, otherwise delay does not exist
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_features,1,bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self,x):
        # forward pass application to record
        for layer in self.hidden_layers:
            x = layer(x)

        return self.output_layer(x)