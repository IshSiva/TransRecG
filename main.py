
import torch
from src.data.process_data import get_data
from src.data.movie_dataset import MovieDataset
from src.config import *

#get the data
get_data()

#create datasets
train_dataset = MovieDataset("src/data/train.csv")
val_dataset = MovieDataset("src/data/validation.csv")
test_dataset = MovieDataset("src/data/test.csv")

#create dataloaders
train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True
        )
val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=VALIDATION_BATCH_SIZE,
            shuffle=True
        )

test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=TEST_BATCH_SIZE,
            shuffle=True
        )




