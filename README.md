# Recommendation Engines

This workshop introduces you to two different approaches for developing recommendation engines. Recommendation engines are widely used machine learning solutions that find items that someone will like based on the past behavior, such as purchasing history. We're all familiar with Netflix recommending movies based on your viewing history and the ratings you've given to other movies. For the first approach, we will delve into how to build these recommender systems by implementing a matrix factorization technique with Apache MXNet. In the second approach, we will utilize SageMaker's built-in algorithm, Object2Vec. For both models, we will be using the MovieLens dataset to build our representation upon. The MovieLens 20M dataset is comprised of movie ratings from the MovieLens site (https://movielens.org/), a site that will predict what other movies you will like after seeing you rate movies. The MovieLens 20M dataset is a sampling of ~20 million ratings from ~138 thousand users on ~27 thousand movies. The ratings range from 0.5 to 5 stars in 0.5 star increments.

## Approach 1: Implementing Deep Matrix Factorization

For the example of movie recommendation, the input takes the form of a matrix where the rows are users, the columns are movies, and the target variable in the matrix is the rating that the user has given the movie. Most of the values in this matrix are missing, as users have only been exposed to a small subset of movies. This is what leads to the challenges associated with sparse input data.

Matrix factorization is a method that learns the connections between the known values in order to infer the missing values. Simply put, for each row and for each column it learns an arbitrary number of numeric factors that represent the row or column. In the example of movie recommendations, the factors for a movie could correspond to "is this a comedy?" or "is a big name actor/actress in this movie?" The factors for a user might correspond to "does this user like comedies?" and "does this user like a given actor/actress?" Ratings are then determined by a dot product between these two smaller matrices (the user factors and the movie factors) such that an individual prediction for how the user will rate movie is calculated. 

Matrix factorization is a linear method, meaning that if there are complicated non-linear interactions going on in the dataset, a simple dot product may not be able to handle it well. A way to better model these non-linearities is "deep matrix factorization" which replaces the dot product with a neural network that is trained jointly with the factors. This makes the model more powerful because a neural network can understand important non-linear combinations of factors to make better predictions.

## Approach 2: Using Object2Vec

Object2Vec is a highly customizable multi-purpose algorithm that can learn embeddings of pairs of objects. The embeddings are learned such that it preserves their pairwise similarities in the original space. Embeddings are an important feature engineering technique in machine learning because they represent discrete variables as continuous vectors to make machine learning on large sparse vector inputs possible. Embeddings capture the semantics of the underlying data by placing similar items close together in the low-dimensional space. This makes the features more effective for building accurate  models. 

In this workshop we demonstrate how Object2Vec can be used to solve problems arising in recommendation systems. Specifically,

We provide the algorithm with (UserID, MovieID) pairs; for each such pair, we also provide a "label" that tells the algorithm whether the user and movie are similar or not. 
1. When the labels are real-valued, we use the algorithm to predict the exact ratings of a movie given a user
1. When the labels are binary, we use the algorithm to recommendation movies to users

The diagram below shows the customization of our model with respect to the problem of predicting movie ratings on inputs that include UserID, ItemID, and Rating. Here, ratings are real-valued

<img src="images/image_ml_rating.png" height="400" width="600">

## Next Steps

Go to [Lab 1: Introduction to Factorization Machines](Lab1%20-%20Introduction%20to%20Factorization%20Machines) 

The first lab contains essential instructions for setting up your SageMaker environment so you can easily work with the Jupyter notebooks that have been provided. Importantly, we setup an end-to-end test system so you can see how your solution works in serverless production. 

The recommended approach to this workshop is to take time to carefully read and understand the theory behind each approach. When you feel comfortable with the concepts execute the notebook code step by step taking care to understand the impact of each statement. 

When you've completed Lab 1, please move on [Lab 2: Introduction to Object2Vec](Lab2%20-%20Introduction%20to%20Object2Vec). 

## Follow Up

**Don't leave this workshop without cleaning up!** You'll find detailed instructions at the end of [Lab 2: Introduction to Object2Vec](Lab2%20-%20Introduction%20to%20Object2Vec) and avoid unnecessary expenses. 

Matrix factorization is a simple approach that is valid for a large number of use cases and is frequently part of a pipeline of larger solutions. ObjectToVec is a powerful approach to datasets with a large number of features. 

Of course there are many other ways to find similarities in your datasets and pair your customers with products. Participants are encouraged to explore shallow learning techniques such as [TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html), and [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) as well as the deep learning techniques presented here. 

Consider how the data you work with every day might fit into these approaches. Your data is unique and you shouldn't expect to cut-and-paste solutions to your unique problems. Your one-of-a-kind data is the most precious asset in your company. Continuously explore that data, experiment with new features, augment your data with public datasets and perhaps even commercial data. Consider using commercial tools such as [Trifacta](https://www.trifacta.com/), [Paxata](https://www.paxata.com/), [TIBCO](https://www.tibco.com/products/data-science), and other[ APN Competency Partners](https://aws.amazon.com/machine-learning/partner-solutions/) for robust data exploration and solution development. 

The real life experiences we've encountered at Amazon have led to some exciting innovations. We hope these labs accelerate innovation and discovery in your practice. 
