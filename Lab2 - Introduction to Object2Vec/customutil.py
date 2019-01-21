import os
import sys
import csv, jsonlines
import numpy as np
import copy
import random

## some utility functions

def load_csv_data(filename, delimiter, verbose=True):
    """
    input: a file readable as csv and separated by a delimiter
    and has format users - movies - ratings - etc
    output: a list, where each row of the list is of the form
    {'in0':userID, 'in1':movieID, 'label':rating}
    """
    to_data_list = list()
    users = list()
    movies = list()
    ratings = list()
    unique_users = set()
    unique_movies = set()
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for count, row in enumerate(reader):
            #if count!=0:
            to_data_list.append({'in0':[int(row[0])], 'in1':[int(row[1])], 'label':float(row[2])})
            users.append(row[0])
            movies.append(row[1])
            ratings.append(float(row[2]))
            unique_users.add(row[0])
            unique_movies.add(row[1])
    if verbose:
        print("In file {}, there are {} ratings".format(filename, len(ratings)))
        print("The ratings have mean: {}, median: {}, and variance: {}".format(
                                            round(np.mean(ratings), 2), 
                                            round(np.median(ratings), 2), 
                                            round(np.var(ratings), 2)))
        print("There are {} unique users and {} unique movies".format(len(unique_users), len(unique_movies)))
    return to_data_list


def csv_to_augmented_data_dict(filename, delimiter):
    """
    Input: a file that must be readable as csv and separated by delimiter (to make columns)
    has format users - movies - ratings - etc
    Output:
      Users dictionary: keys as user ID's; each key corresponds to a list of movie ratings by that user
      Movies dictionary: keys as movie ID's; each key corresponds a list of ratings of that movie by different users
    """
    to_users_dict = dict() 
    to_movies_dict = dict()
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for count, row in enumerate(reader):
            #if count!=0:
            if row[0] not in to_users_dict:
                to_users_dict[row[0]] = [(row[1], row[2])]
            else:
                to_users_dict[row[0]].append((row[1], row[2]))
            if row[1] not in to_movies_dict:
                to_movies_dict[row[1]] = list(row[0])
            else:
                to_movies_dict[row[1]].append(row[0])
    return to_users_dict, to_movies_dict


def user_dict_to_data_list(user_dict):
    # turn user_dict format to data list format (acceptable to the algorithm)
    data_list = list()
    for user, movie_rating_list in user_dict.items():
        for movie, rating in movie_rating_list:
            data_list.append({'in0':[int(user)], 'in1':[int(movie)], 'label':float(rating)})
    return data_list

def divide_user_dicts(user_dict, sp_ratio_dict):
    """
    Input: A user dictionary, a ration dictionary
         - format of sp_ratio_dict = {'train':0.8, "test":0.2}
    Output: 
        A dictionary of dictionaries, with key corresponding to key provided by sp_ratio_dict
        and each key corresponds to a subdivded user dictionary
    """
    ratios = [val for _, val in sp_ratio_dict.items()]
    assert np.sum(ratios) == 1, "the sampling ratios must sum to 1!"
    divided_dict = {}
    for user, movie_rating_list in user_dict.items():
        sub_movies_ptr = 0
        sub_movies_list = []
        #movie_list, _ = zip(*movie_rating_list)
        #print(movie_list)
        for i, ratio in enumerate(ratios):
            if i < len(ratios)-1:
                sub_movies_ptr_end = sub_movies_ptr + int(len(movie_rating_list)*ratio)
                sub_movies_list.append(movie_rating_list[sub_movies_ptr:sub_movies_ptr_end])
                sub_movies_ptr = sub_movies_ptr_end
            else:
                sub_movies_list.append(movie_rating_list[sub_movies_ptr:])
        for subset_name in sp_ratio_dict.keys():
            if subset_name not in divided_dict:
                divided_dict[subset_name] = {user: sub_movies_list.pop(0)}
            else:
                #access sub-dictionary
                divided_dict[subset_name][user] = sub_movies_list.pop(0)
    
    return divided_dict

def write_csv_to_jsonl(jsonl_fname, csv_fname, csv_delimiter):
    """
    Input: a file readable as csv and separated by delimiter (to make columns)
        - has format users - movies - ratings - etc
    Output: a jsonline file converted from the csv file
    """
    with jsonlines.open(jsonl_fname, mode='w') as writer:
        with open(csv_fname, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=csv_delimiter)
            for count, row in enumerate(reader):
                #print(row)
                #if count!=0:
                writer.write({'in0':[int(row[0])], 'in1':[int(row[1])], 'label':float(row[2])})
        print('Created {} jsonline file'.format(jsonl_fname))
                    
    
def write_data_list_to_jsonl(data_list, to_fname):
    """
    Input: a data list, where each row of the list is a Python dictionary taking form
    {'in0':userID, 'in1':movieID, 'label':rating}
    Output: save the list as a jsonline file
    """
    with jsonlines.open(to_fname, mode='w') as writer:
        for row in data_list:
            #print(row)
            writer.write({'in0':row['in0'], 'in1':row['in1'], 'label':row['label']})
    print("Created {} jsonline file".format(to_fname))

def data_list_to_inference_format(data_list, binarize=True, label_thres=3):
    """
    Input: a data list
    Output: test data and label, acceptable by SageMaker for inference
    """
    data_ = [({"in0":row['in0'], 'in1':row['in1']}, row['label']) for row in data_list]
    data, label = zip(*data_)
    infer_data = {"instances":data}
    if binarize:
        label = get_binarized_label(list(label), label_thres)
    return infer_data, label


def get_binarized_label(data_list, thres):
    """
    Input: data list
    Output: a binarized data list for recommendation task
    """
    for i, row in enumerate(data_list):
        if type(row) is dict:
            #if i < 10:
                #print(row['label'])
            if row['label'] > thres:
                #print(row)
                data_list[i]['label'] = 1
            else:
                data_list[i]['label'] = 0
        else:
            if row > thres:
                data_list[i] = 1
            else:
                data_list[i] = 0
    return data_list

def get_class_accuracy(res, labels, thres):
    if type(res) is dict:
        res = res['predictions']
    assert len(res)==len(labels), 'result and label length mismatch!'
    accuracy = 0
    for row, label in zip(res, labels):
        if type(row) is dict:
            if row['scores'][1] > thres:
                prediction = 1
            else: 
                prediction = 0
            if label > thres:
                label = 1
            else:
                label = 0
            accuracy += 1 - (prediction - label)**2
    return accuracy / float(len(res))

def get_movie_embedding_dict(movie_ids, trained_model):
    input_instances = list()
    for s_id in movie_ids:
        input_instances.append({'in1': [s_id]})
    data = {'instances': input_instances}
    movie_embeddings = trained_model.predict(data)
    embedding_dict = {}
    for s_id, row in zip(movie_ids, movie_embeddings['predictions']):
        embedding_dict[s_id] = np.array(row['embeddings'])
    return embedding_dict


def load_movie_id_name_map(item_file):
    movieID_name_map = {}
    with open(item_file, 'r', encoding="ISO-8859-1") as f:
        for row in f.readlines():
            row = row.strip()
            split = row.split('|')
            movie_id = split[0]
            movie_name = split[1]
            sparse_tags = split[-19:]
            movieID_name_map[int(movie_id)] = movie_name 
    return movieID_name_map

            
def get_nn_of_movie(movie_id, candidate_movie_ids, embedding_dict):
    movie_emb = embedding_dict[movie_id]
    min_dist = float('Inf')
    best_id = candidate_movie_ids[0]
    for idx, m_id in enumerate(candidate_movie_ids):
        candidate_emb = embedding_dict[m_id]
        curr_dist = np.linalg.norm(candidate_emb - movie_emb)
        if curr_dist < min_dist:
            best_id = m_id
            min_dist = curr_dist
    return best_id, min_dist


def get_unique_movie_ids(data_list):
    unique_movie_ids = set()
    for row in data_list:
        unique_movie_ids.add(row['in1'][0])
    return list(unique_movie_ids)