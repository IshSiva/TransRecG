# TransRecG: Transformer Based Sequential Recommendation System with Graph Embeddings

# Project Overview 
In recent years, Deep Learning-based Recommendation Systems (DLRS) have become the industry standard. However, many existing DLRS ignore the temporal information that 
can be extracted from the user's history of interactions. They also fail to capture the higher order connectivity features like user-user and movie-movie(or item) 
relations. To address the above problems, we propose TransRecG: a Transformer-based Recommendation System using Graph Embeddings. First, we create a user-movie-attribute
knowledge graph (KG). A Neural Graph Collaborative Filtering (NGCF) model that uses a Relational Graph Convolutional Neural Network (RGCN) is trained on this KG to 
capture the user-movie links and high order user-user and movie-movie relations. Second, we use a Transformer to understand the user's behavior based on the sequential
order of the movies watched by them. We use embeddings learnt by the RGCN model to initialize the user and movie embeddings for the Transformer. Therefore, our proposed
TransRecG model captures both the higher order connectivity information, and the sequential order with which the users interact with the movies. The model predicts the
rating a user will give for a movie based on their watch history. The movie with the highest rating can be recommended as the next movie that the user will watch.
We analyze the performance of our model on the MoviesLens1M dataset which contains 6,040 users and 3,883 movies, and show that the proposed model outperforms existing
baselines. We also demonstrate how the model efficiently handles cold start users, and how the number of samples in the training dataset affect the model's performance.  
