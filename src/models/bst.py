import torch
import torch.nn as nn
import math


from src.models.positional_embedding import PositionalEmbedding
from src.config import *


class BST(nn.Module):
    def __init__(self, users, movies, device):
        super().__init__()
        self.device = device
        
        genres = [
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]

        # Embedding layers

        # Users embedding using the user id and metadata like sex, age group, occupation and zip code
        self.embeddings_user_id = nn.Embedding(
            int(users.user_id.max())+1, int(math.sqrt(users.user_id.max()))+1
        )
        
        # Users metadata embeddings
        self.embeddings_user_sex = nn.Embedding(
            len(users.sex.unique()), int(math.sqrt(len(users.sex.unique())))
        )
        self.embeddings_age_group = nn.Embedding(
            len(users.age_group.unique()), int(math.sqrt(len(users.age_group.unique())))
        )
        self.embeddings_user_occupation = nn.Embedding(
            len(users.occupation.unique()), int(math.sqrt(len(users.occupation.unique())))
        )
        self.embeddings_user_zip_code = nn.Embedding(
            len(users.zip_code.unique()), int(math.sqrt(len(users.sex.unique())))
        )
        
        # Movies
        self.embeddings_movie_id = nn.Embedding(
            int(movies.movie_id.max())+1, int(math.sqrt(movies.movie_id.max()))+1
        )
        
        # Movies features embeddings using genres
        genre_vectors = movies[genres].to_numpy()
        self.embeddings_movie_genre = nn.Embedding(
            genre_vectors.shape[0], genre_vectors.shape[1]
        )
        
           
        self.embeddings_movie_year = nn.Embedding(
            len(movies.year.unique()), int(math.sqrt(len(movies.year.unique())))
        )
        
        self.positional_embedding = PositionalEmbedding(8, 63)
        
        # Network
        self.transfomerlayer = nn.TransformerEncoderLayer(63, 3, dropout=0.2)
        self.linear = nn.Sequential(
            nn.Linear(
                589,
                1024,
            ),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )
        self.criterion = torch.nn.MSELoss()
       
        self.opt = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE)
        


    def encode_input(self,inputs):
        inputs = [x.to(self.device) for x in inputs]
        user_id, movie_history, target_movie_id,  movie_history_ratings, target_movie_rating, sex, age_group, occupation = inputs
               
        #MOVIES
        movie_history = self.embeddings_movie_id(movie_history)
        target_movie = self.embeddings_movie_id(target_movie_id)
         
        target_movie = torch.unsqueeze(target_movie, 1)
        transfomer_features = torch.cat((movie_history, target_movie),dim=1)

        #USERS
        user_id = self.embeddings_user_id(user_id)
        
        sex = self.embeddings_user_sex(sex)
        age_group = self.embeddings_age_group(age_group)
        occupation = self.embeddings_user_occupation(occupation)
        user_features = torch.cat((user_id, sex, age_group,occupation), 1)
        
        return transfomer_features, user_features, target_movie_rating.float(), movie_history_ratings
    
    def forward(self, batch):
        transfomer_features, user_features, target_movie_rating, movie_history_ratings = self.encode_input(batch)
        positional_embedding = self.positional_embedding(transfomer_features)
        
        transfomer_features = transfomer_features + positional_embedding        
        movie_history_ratings = torch.concat((movie_history_ratings, 5 * torch.ones(movie_history_ratings.size()[0], 1).to(self.device)), 1)[:, :, None]/5.0

        transfomer_features = transfomer_features * movie_history_ratings

        transformer_output = self.transfomerlayer(transfomer_features)
        transformer_output = torch.flatten(transformer_output,start_dim=1)
        
        #Concat with other features
        features = torch.cat((transformer_output,user_features),dim=1)

        output = self.linear(features)
        return output, target_movie_rating
