import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import os
import pandas as pd



s3_path = 'your-s3-uri-to-the-csv'

df = pd.read_csv(s3_path, sep='\t',names=['ID','TITLE','URL','PUBLISHER','CATEGORY','STORY','HOSTNAME','TIMESTAMP'])

df = df[['TITLE','CATEGORY']]



my_dict = {
    'e':'Entertainment',
    'b':'Business',
    't':'Science',
    'm':'Health'
}

def update_cat(x):
    return my_dict[x]

df['CATEGORY'] = df['CATEGORY'].apply(lambda x:update_cat(x))



# This is just a tip
#df = df.sample(frac=0.05,random_state=1)

#df = df.reset_index(drop=True)
#This is where the tip ends

encode_dict = {}



def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x]=len(encode_dict)
    return encode_dict[x]

df['ENCODE_CAT']= df['CATEGORY'].apply(lambda x:encode_cat(x))

# resets the index of a Dataframe, which mens it creatse a new range index starting from 0 to len(df)-1, old indexes are dropped
df = df.reset_index(drop=True)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
         # uses integer-based indexing meaning index is treated as a position in the df. 0 selects the row at the specified index, TITLE column
        title = str(self.data.iloc[index, 0])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.iloc[index, 2], dtype=torch.long) # same thing with iloc here, 2 means we access column 3
        }

    def __len__(self):
        return self.len



train_size = 0.8
train_dataset = df.sample(frac=train_size,random_state=200)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)

train_dataset.reset_index(drop=True)


print("Full dataset: {}".format(df.shape))
print("Train dataset: {}".format(train_dataset.shape))
print("Test dataset: {}".format(test_dataset.shape))


MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2



training_set = NewsDataset(train_dataset,tokenizer,MAX_LEN)
testing_set = NewsDataset(test_dataset,tokenizer,MAX_LEN)


train_parameters = {
                    'batch_size':TRAIN_BATCH_SIZE,
                    'shuffle':True,
                    'num_workers':0
                    }
test_parameters = {
                    'batch_size':VALID_BATCH_SIZE,
                    'shuffle':True,
                    'num_workers':0
                    }


training_loader = DataLoader(training_set, **train_parameters)
testing_loader = DataLoader(testing_set, **test_parameters)