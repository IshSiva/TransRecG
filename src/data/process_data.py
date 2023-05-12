from urllib.request import urlretrieve
from zipfile import ZipFile
import pandas as pd

import sys
sys.path.append("../..")

from src.config import SEQUENCE_LENGTH, STEP_SIZE, TRAIN_SPLIT, VALIDATION_SPLIT


def get_data_from_source():
    """
    get the MovieLens1M dataset zip file from the remote source.

    Returns:
        None
    
    The zipfile is downloaded in the data/ folder.
    """
    target_dir = "src/data/"
    urlretrieve("http://files.grouplens.org/datasets/movielens/ml-1m.zip", "src/data/movielens.zip")
    ZipFile("src/data/movielens.zip", "r").extractall(target_dir)


def preprocess_data():
    """
    Performs preprocessing on the raw data from MovieLens. Returns users, movies and ratings dataframes.

    Returns:
        users(PandasDataframe) : df of user details.
        movies(PandasDataframe) : df of movie details.
        ratings(PandasDataframe) : df of user-movie rating details.
    """

    users = pd.read_csv(
    "src/data/ml-1m/users.dat",
    sep="::",
    engine='python',
    names=["user_id", "sex", "age_group", "occupation", "zip_code"],
    )

    ratings = pd.read_csv(
        "src/data/ml-1m/ratings.dat",
        sep="::",
        engine='python',
        names=["user_id", "movie_id", "rating", "unix_timestamp"],
    )

    movies = pd.read_csv(
        "src/data/ml-1m/movies.dat", 
        sep="::", 
        names=["movie_id", "title", "genres"],
        engine='python',
        encoding="ISO-8859-1"
    )

    ## Movies
    movies["year"] = movies["title"].apply(lambda x: x[-5:-1])
    movies.year = pd.Categorical(movies.year)
    movies["year"] = movies.year.cat.codes

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

    for genre in genres:
        movies[genre] = movies["genres"].apply(
            lambda values: int(genre in values.split("|"))
    )

    ## Users
    users.sex = pd.Categorical(users.sex)
    users["sex"] = users.sex.cat.codes

    users.age_group = pd.Categorical(users.age_group)
    users["age_group"] = users.age_group.cat.codes

    users.occupation = pd.Categorical(users.occupation)
    users["occupation"] = users.occupation.cat.codes

    users.zip_code = pd.Categorical(users.zip_code)
    users["zip_code"] = users.zip_code.cat.codes

    return users, movies, ratings


def create_ratings_df(ratings, user_df):

  """
    Creates sequences of input from the original ratings dataframe
    
    Args:
      ratings(PandasDataframe) : A dataframe of user's ratings for movies. 
      user_df(PandasDataframe) : A dataframe of user details

    Returns:
      ratings_data_transformed(PandasDataframe) : A pandas dataframe that contains user_id corresponding sequence of movies that 
        the user has reviewed. The sequences are of length SEQUENCE_LENGTH.
  """

  ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")

  ratings_data = pd.DataFrame(
      data={
          "user_id": list(ratings_group.groups.keys()),
          "movie_ids": list(ratings_group.movie_id.apply(list)),
          "ratings": list(ratings_group.rating.apply(list)),
          "timestamps": list(ratings_group.unix_timestamp.apply(list)),
      }
  )

  def create_sequences(values, window_size, step_size):
      sequences = []
      start_index = 0
      while True:
          end_index = start_index + window_size
          if end_index<len(values):
            seq = values[start_index:end_index]
            sequences.append(seq)
          else:
            break
          start_index += step_size

      return sequences

  #create sequences of movie ids
  ratings_data.movie_ids = ratings_data.movie_ids.apply(
    lambda ids: create_sequences(ids, SEQUENCE_LENGTH, STEP_SIZE)
  )

  #create corresponding sequences of movie ratings
  ratings_data.ratings = ratings_data.ratings.apply(
      lambda ids: create_sequences(ids, SEQUENCE_LENGTH, STEP_SIZE)
  )

  ratings_data_movies = ratings_data[["user_id", "movie_ids"]].explode(
    "movie_ids", ignore_index=True
  )
  ratings_data_rating = ratings_data[["ratings"]].explode("ratings", ignore_index=True)
  ratings_data_transformed = pd.concat([ratings_data_movies, ratings_data_rating], axis=1)
  ratings_data_transformed = ratings_data_transformed.join(
    user_df.set_index("user_id"), on="user_id"
  )

  del ratings_data_transformed["zip_code"]

  ratings_data_transformed.rename(
    columns={"movie_ids": "sequence_movie_ids", "ratings": "sequence_ratings"},
    inplace=True,
  )

  ratings_data_transformed = ratings_data_transformed.dropna()

  return ratings_data_transformed


def split_data_masks(ratings):
  """
  This project divides the data based on timestamp. This method has been chosen over random split to avoid data leakage.
  
  Args:
      ratings(PandasDataframe) : A dataframe of user's ratings for movies. 

  Returns:
      train_mask(PandasSeries): A mask of the data that's to be selected for training
      validation_mask(PandasSeries): A mask of the data that's to be selected for validation
      test_mask(PandasSeries): A mask of the data that's to be selected for testing
  """

  train_timestamp = ratings["unix_timestamp"].quantile(TRAIN_SPLIT)
  val_timestamp = ratings["unix_timestamp"].quantile(TRAIN_SPLIT+VALIDATION_SPLIT)
  
  train_mask = ratings["unix_timestamp"]<train_timestamp
  val_mask = (ratings["unix_timestamp"]>train_mask)&(ratings["unix_timestamp"]<val_timestamp)
  test_mask = (ratings["unix_timestamp"]>val_timestamp)

  return train_mask, val_mask, test_mask


def get_data():
    """
    driver function to get the raw data, process the data and generate train, validation and test sets.

    """

    #fetch the data from remote url
    get_data_from_source()
    
    #pre-process the data
    user, movies, ratings = preprocess_data()
    
    #data masks
    train_mask, val_mask, test_mask = split_data_masks(ratings)

    #split the data
    #training set
    train_ratings = ratings[train_mask]
    train_data = create_ratings_df(train_ratings, user)
    train_data.to_csv("src/data/train.csv")

    #validation set
    validation_ratings = ratings[val_mask]
    validation_data = create_ratings_df(validation_ratings, user)
    validation_data.to_csv("src/data/validation.csv")

    #test set
    test_ratings = ratings[test_mask]
    test_data = create_ratings_df(test_ratings, user)
    test_data.to_csv("src/data/test.csv")





   
    



    
