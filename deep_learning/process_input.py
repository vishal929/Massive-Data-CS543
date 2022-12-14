# file is for processing our data which was processed by pyspark for task 1
# goal is to modify the data so that it is ready for prediction task

# categorical task: predict if there is delay or not (1 or 0)
# regression task: predict how much of a delay there is

# pandas to work with parquets
import pandas as pd
# pytorch dataset is our desired output
import torch
from torch.utils.data import Dataset
# sklearn train_test_split will be useful for initially splitting the data
from sklearn.model_selection import train_test_split


# function to modify the depDelay column to align with requirements for label (categorical) prediction
def modify_dep_delay(depDelay):
    if depDelay > 0:
        # there was a delay
        return 1
    else:
        return 0


# this creates train,test,and validation splits for both categorical tasks and the regression task
# is_categorical is True if we want to process for the categorical task, otherwise false
def process_dataset(is_categorical):
    # reading in multiple parquet files created by spark (since they were written in partitions)
    df = pd.read_parquet('../FINAL_processed_data')
    # applying function to "DepDelay" column to make it either a 1 if there is a delay, or 0 if there is no delay
    if is_categorical:
        df['DepDelay'] = df['DepDelay'].map(modify_dep_delay)

    # splitting into train_test_val
    train, remaining = train_test_split(df, train_size=0.8, shuffle=True)
    # splitting the remaining examples in half to create validation and test sets
    test, validation = train_test_split(remaining, train_size=0.5)

    # saving the data splits to parquet
    if is_categorical:
        train.to_parquet('./categorical_train')
        test.to_parquet('./categorical_test')
        validation.to_parquet('./categorical_validation')
    else:
        train.to_parquet('./regression_train')
        test.to_parquet('./regression_test')
        validation.to_parquet('./regression_validation')


# creating a custom pytorch dataset for our tasks
class Airplane_Weather_Dataset(Dataset):
    # standard transform is max min
    def __init__(self, task, split):
        super(Airplane_Weather_Dataset, self).__init__()
        # we need to normalize by train data, not validation or test data (or else we are "leaking" information)
        train_records = pd.read_parquet(task+'_'+'train')
        self.records = pd.read_parquet(task + '_' + split)

        # dropping ActualElapsedTime column because that is cheating (looking into future)
        # also dropping record_id column
        train_records.drop(columns=['ActualElapsedTime','record_id'],inplace=True)
        self.records.drop(columns=['ActualElapsedTime','record_id'],inplace=True)
        self.records_max = train_records.max()
        self.records_min = train_records.min()

        # min-max scaling
        self.records = (self.records - self.records_min) / (self.records_max - self.records_min)

        # turning data into tensors ready for training
        self.labels = torch.tensor(self.records['DepDelay'].to_numpy(),dtype=torch.float32)
        self.records = self.records.drop(columns=['DepDelay']).to_numpy()
        self.records = torch.squeeze(torch.tensor(self.records,dtype=torch.float32))


    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        # returning (processed_record, label/target) for prediction tasks
        return self.records[idx],self.labels[idx]