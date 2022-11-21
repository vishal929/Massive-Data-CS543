# this file defines some standard modules we will use in our categorical and regression modules

import torch

class FeedForward(torch.nn.Module):
    # we need to provide the number in the batch for batch normalization
    def __init__(self,input_features,output_features):
        # feedforward consists of linear layer followed by batchnorm, followed by
        # a hidden layer followed by some dropout, we will use GeLu
        super(FeedForward, self).__init__()
        # bias is set to true by default
        self.linear = torch.nn.Linear(input_features,output_features,bias=True)
        self.batchnorm = torch.nn.BatchNorm1d(output_features)
        # gelu as the activation function  (multiplies input by CDF(input) for standard gaussian)
        self.gelu = torch.nn.GELU()
        # dropout as a regularization scheme
        self.dropout = torch.nn.Dropout(p=0.2)
        self.layer = torch.nn.Sequential(
            self.linear,
            self.batchnorm,
            self.gelu,
            self.dropout
        )

    def forward(self,x):
        # just apply these layers in sequence
        return self.layer(x)
