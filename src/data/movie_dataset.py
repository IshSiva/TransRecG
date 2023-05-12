import torch.utils.data as data
import torch
import pandas as pd

class MovieDataset(data.Dataset):
    """Movie dataset."""

    def __init__(
        self, ratings_file,test=False
    ):
        """
        Args:
            csv_file (string): Path to the csv file with user,past,future.
        """
        self.ratings_frame = pd.read_csv(
            ratings_file,
            delimiter=",",
            # iterator=True,
        )
        self.test = test

    def __len__(self):
        return len(self.ratings_frame)

    def __getitem__(self, idx):
        data = self.ratings_frame.iloc[idx]
        user_id = data.user_id
        
        movie_history = eval(data.sequence_movie_ids)
        movie_history_ratings = eval(data.sequence_ratings)
        target_movie_id = movie_history[-1:][0]
        target_movie_rating = movie_history_ratings[-1:][0]
        
        movie_history = torch.LongTensor(movie_history[:-1])
        movie_history_ratings = torch.LongTensor(movie_history_ratings[:-1])
        
        sex = data.sex
        age_group = data.age_group
        occupation = data.occupation
        
        return user_id, movie_history, target_movie_id,  movie_history_ratings, target_movie_rating, sex, age_group, occupation