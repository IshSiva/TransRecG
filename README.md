# TransRecG: Transformer Based Sequential Recommendation System with Graph Embeddings

# Project Overview 
In recent years, Deep Learning-based Recommendation Systems (DLRS) have become the industry standard. However, many existing DLRS ignore the temporal information that 
can be extracted from the user's history of interactions. They also fail to capture the higher order connectivity features like user-user and movie-movie(or item) 
relations. 

To address the above problems, we propose TransRecG: a Transformer-based Recommendation System using Graph Embeddings. First, we create a user-movie-attribute
knowledge graph (KG). A Neural Graph Collaborative Filtering (NGCF) model that uses a Relational Graph Convolutional Neural Network (RGCN) is trained on this KG to 


capture the user-movie links and high order user-user and movie-movie relations. Second, we use a Transformer to understand the user's behavior based on the sequential
order of the movies watched by them. We use embeddings learnt by the RGCN model to initialize the user and movie embeddings for the Transformer. Therefore, our proposed
TransRecG model captures both the higher order connectivity information, and the sequential order with which the users interact with the movies. The model predicts the
rating a user will give for a movie based on their watch history. The movie with the highest rating can be recommended as the next movie that the user will watch.

We analyze the performance of our model on the MoviesLens1M dataset which contains 6,040 users and 3,883 movies, and show that the proposed model outperforms existing
baselines. We also demonstrate how the model efficiently handles cold start users, and how the number of samples in the training dataset affect the model's performance.



# Baselines

## 1. Neural Graph Collaborative Filtering
The NGCF [1] model utilizes a GCN to generate node embeddings from the user-movie bipartite graph. There exists an edge between the user and the movie, if the user has rated the movie. All nodes are initialized with the adjacency matrix representation. We use a 2 layer message passing GCN with an hidden dimension of 18, for generating the user and movie embeddings. For all ratings in the dataset, we concatenate the corresponding user and movie embeddings, and pass it through a MLP for predicting the rating.


## 2. Behaviour Sequence Transformers

![bst](https://user-images.githubusercontent.com/34617363/235417917-1aed4da8-6bfa-4e75-892b-52cf8fca0641.JPG)


The BST [2] model uses a transformer encoder-based architecture for predicting the rating that the a given user will provide for $n^{th}$ movie given the previous $n-1$ movies the user has reviewed. In this project, we consider $n = 8$. The model generates user and movie embeddings of size 63 using their metadata. A single multi-head attention layer with 9 attention heads is used for capturing the temporal relations between various movies. Before the movie embeddings are passed to the Transformer, positional encoding is added to them to preserve the ordering, and the resulting embeddings are multiplied by their corresponding ratings. The output of the Transformer is then concatenated with the user embeddings, and are then passed to a MLP for predicting the ratings. 


# TransRecG

![trans](https://user-images.githubusercontent.com/34617363/235418423-2afcc5df-c39a-43d7-97cc-6bc70750239e.png)


The NGCF model does not consider the sequential ordering of the movies, whereas the BST model fails to capture higher order user-user or movie-movie relations. The TransRecG model overcomes these drawbacks by combining the approaches described in the NGCF and BST models. The proposed TransRecG model has 3 main components: (1) the user-movie-attribute KG, (2) NGCF using RGCN, and (3) BST.

## Knowledge Graph Construction
The KG contains a node for all *user\_id*, *movie\_id* and *genre*. The KG is composed of two bipartite graphs. (1) The **user-movie graph**: if the user has rated a movie then there exist an edge between the user and the movie. (2) The **movie-attribute graph**: if the movie belongs to a particular genre then there exist an edge between the genre and movie. As a single movie can have multiple genres, there can exist more than one edge for a movie in the movie-attribute graph. 

The KG is an heterogeneous graph in terms of it's edges. There are 5 different edge types in the user-movie subgraph, and the edge type between a particular user and a movie corresponds to the rating (1-5) the user has given to that movie. All edges in the movie-attribute subgraph belongs to the same type. Here, we consider *genre* as the only attribute for the movies, and all the edges belong to the same type ''belongs to''.

We initialize the *movie\_id* and *genre* node embeddings with their corresponding GloVe representations (dimension = 50). For example, if the *genre* is ''Comedy,'' then the node is initialized with the GloVE embedding of the word ''Comedy''. The user embeddings are randomly initialized. 

Using a KG for generating embeddings helps in capturing meaningful user-user and movie-movie relations which helps with personalization in the recommendation tasks. As similar users tend to watch similar movies, the similarities between users and movies can be captured using the KG.

## NGCF on KG
<img width="835" alt="kg" src="https://user-images.githubusercontent.com/34617363/235418440-0a790846-40ec-42c2-87b2-c24514fbe9c6.png">

Similar to the baseline model, we train an RGCN model (instead of a GCN) for generating the user and movie embeddings from the KG using the NGCF framework. The RGCN model learns different sets of weights for different edge types, instead of learning one set of weights for all the edges. By utilizing an RGCN, the model is able to reason more effectively about the complex relationships between entities in the Knowledge Graph, leading to improved performance on tasks that require relational reasoning. In particular, this will eventually help with the user cold-start problem by allowing us to explore movie-movie relations as we have a lack of user-user and user-movie information as we will see in our final results. For this project, we use a 2 layer message passing RGCN model, with hidden dimensions of size 50, trained on the rating prediction task. We use the AdamW optimizer and the RMSE Loss for optimizing the model parameters. 

## BST using Graph Embeddings
The architecture of the BST is similar to that of the baseline BST. However, in TransRecG model the user and movie embeddings are initialized with their corresponding node embeddings from the NGCF. Therefore, the benefits are two fold: (1) The model can adapt to the varying preferences of the user by analysing the previous movies the user has watched, and finding similar movies using the movie embeddings learnt from NGCF. For example, a user who has watched ''Harry Potter 1'' (*HP1*) could be recommended *HP2*, as many users watch *HP2* after *HP1* (from BST model), and the movies *HP1* and *HP2* are very similar (from NGCF model). (2) The model can personalize recommendations by using the user embeddings to understand user preferences. For example, some users might have watched *HP1* and *HP2* because they prefer watching fiction, whereas other users might have watched it for ''Daniel Radcliffe.''

![bst_ngcf](https://user-images.githubusercontent.com/34617363/235418472-a70421f8-0e3b-4533-bf6e-7c27d3750eb0.png)

Node embeddings from the NGCF model of size 50 are passed through a linear layer to get an embedding of size 36. The node embeddings are then concatenated with the BST embeddings learnt from user/movie metadata, before being passed to the BST. The Transformer uses 9 attention heads to learn temporal information from the movie embeddings. The output of the the Transformer model is then merged with the user embeddings before being passed to a 5 layer MLP with layers of size 913, 1024, 512, 256 and 1. The output of the MLP is the predicted rating, and is the output of the TransRecG model. The BST and MLP model parameters are optimized on this rating prediction task using an RMSE Loss, AdamW optimizer with a learning rate of 5e-4 and a batch size of 256. 

In summary, the TransRecG model is able to cater to the varying preferences of the users, and also provide personalized recommendations, by capturing sequential, and higher-order relations between users and movies. 


# Results and Analysis

![table](https://user-images.githubusercontent.com/34617363/235418805-34bc963f-4570-4584-86c1-05197e7443d4.JPG)

- Sequence based models (GRU and BST) perform better than non-sequence based models (NGCF). Even the baseline GRU model outperforms the NGCF techniques. Hence, we can conclude that sequential recommendation systems are more powerful that graph based model, especially when initial knowledge about the graph entites are limited. 
- While analysing the impact of sequence lenghts on the BST model performance, we found that longer sequences lead to better performance. 
- Both the graph embedding augmented BST models (BST-8 + NGCF-BP and TransRecG) outperformed all BST and NGCF models. Therefore, higher order user-user, and movie-movie connectivity information helps in improving recommendations. 
- TransRecG with graph embeddings from RGCN + KG performs better than GCN + BP models. The nodes are initialized with the adjacency matrix in the GCN + BP models, whereas the nodes are initialized using GloVE embeddings in the RGCN + KG model. Therefore, more meaningful node initializations lead to better performance. 

## Cold Start Users

![cold_start_users](https://user-images.githubusercontent.com/34617363/235418479-11855cda-9379-4d86-b8ac-cfbf0796014d.png)

The comparison of the loss of the models for the users having cold start problem is shown above. The dotted lines show the mean loss of the model. The NGCF model considers one user and one movie at a time during the rating prediction task. As there exist no information about cold start users in the training dataset, the model does not learn any meaningful user embeddings. Therefore, during the test time the NGCF model has the worst performance. The sequence based BST models perform much better than the NGCF model, as they are able to leverage the sequential information given as input to generate good recommendations, even without proper user embeddings. 

Adding graph embeddings to the BST model further reduces the loss as movie-movie relations are learnt through the 2 layer message passing network, and this helps in improving the recommendations. Interestingly, the proposed TransRecG model achieves the best performance as it uses node embeddings learnt from the KG which are initialized using GloVE embeddings.

## Non-Cold Start Users (Train Vs Test)

![no_cold_start](https://user-images.githubusercontent.com/34617363/235418461-38962c91-380a-46cb-8f29-6c4699c44428.png)

The comparison of the loss of the models for the non cold start users is shown above. In this figure, the number of samples denotes the number of data points the user has in the training set. The lines represent the trends of the loss of the various models. We observe that the NGCF loss increases for users with more data points in the training dataset. As users with large number of data points tend to have varied preferences, and as the NGCF model fails to capture the dynamic preferences of the users (no sequential information), the test loss increases for users with more traning data points. 

In contrast, for the BST model the test loss decreases with increase in training data points. This shows that the BST model is able to capture the dyanmic preferences of the user, and with more data points its able to learn more about sequential order of the movies, as well as the users. However, the performance of the BST models augmented with graph embeddings almost remains unchanged. As we directly concatenate the user and movie embeddings from the poorly performing NGCF model (for users with large number of data points), this affects the performance of our models. We believe that adding an attention based model for weighting the graph and metadata based embeddings before concatenating them can improve the performance of the model. 

# Future Works

- Currently, the BST model uses a sequence length of 8 due to limited compute. For upto sequence length of 15, we found that model performance improves with increasing sequence lengths. However, the effect of even longer sequence lengths should be further analysed. 
- We implement the Knowledge Graph (KG) as a heterogeneous graph only in terms of the edges. Creating the KG as a fully heterogeneous graph with both different types of nodes and edges, is a relevant future work as user-movie-attribute graphs are composed of different types of entities in nature. 
- Understanding the effect of TransRecG model performance with attention weights for balancing between the graph and meta data embeddings, especially for users with large number of training data points. 

# References 
[1] Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. 2019.
Neural Graph Collaborative Filtering. In Proceedings of the 42nd International ACM
SIGIR Conference on Research and Development in Information Retrieval. 165–174.

[2] Qiwei Chen, Huan Zhao, Wei Li, Pipei Huang, and Wenwu Ou. 2019. Behavior
sequence transformer for e-commerce recommendation in alibaba. In Proceedings
of the 1st International Workshop on Deep Learning Practice for High-Dimensional
Sparse Data. 1–4.
