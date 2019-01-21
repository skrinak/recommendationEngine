import json
import sys, os
import csv, jsonlines
import boto3
import numpy as np

# grab environment variables for model endpoint name
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
# declare and initiate global variable for SageMaker runtime and s3
runtime= boto3.client('runtime.sagemaker')
s3 = boto3.resource('s3')

def lambda_handler(event, context):
    
    data = json.loads(json.dumps(event))
	
	# Initialize constants 
    BUCKET_NAME = data['bucket_name']
    MOVIE_ID_TO_EXAMINE = int(data['movie_id_to_examine']) 
        
    DATADIR='object2vec/movielens/ml-100k'
    TRAIN_DATA_FILE = 'ua.base'
    MOVIE_LOOKUP_FILE = 'u.item'
    UNIQUE_MOVIE_IDS_FILE = 'unique_movie_ids.txt'
    DATA_FILE_KEY = DATADIR + '/' + TRAIN_DATA_FILE
    MOVIE_LOOKUP_KEY = DATADIR + '/' + MOVIE_LOOKUP_FILE
    UNIQUE_MOVIE_IDS_KEY = DATADIR + '/' + UNIQUE_MOVIE_IDS_FILE
    LOCAL_DATA_FILE = '/tmp/' +  TRAIN_DATA_FILE
    LOCAL_MOVIE_LOOKUP_FILE = '/tmp/' +  MOVIE_LOOKUP_FILE
    LOCAL_UNIQUE_MOVIE_IDS_FILE =  '/tmp/' +  UNIQUE_MOVIE_IDS_FILE
	
	# get data files from S3 bucket
    s3.Object(BUCKET_NAME, DATA_FILE_KEY).download_file(LOCAL_DATA_FILE)
    s3.Object(BUCKET_NAME, MOVIE_LOOKUP_KEY).download_file(LOCAL_MOVIE_LOOKUP_FILE)
	
    prefix = 'ml-100k'
    train_data_list = load_csv_data(LOCAL_DATA_FILE, '\t', verbose=False)
    
	# get unique movie Ids from the dataset
    unique_movie_ids = get_unique_movie_ids(train_data_list)
    
    with open(LOCAL_UNIQUE_MOVIE_IDS_FILE, 'w') as outfile:
        json.dump(unique_movie_ids, outfile)

    s3.meta.client.upload_file(LOCAL_UNIQUE_MOVIE_IDS_FILE, BUCKET_NAME, UNIQUE_MOVIE_IDS_KEY)
    
	# get movie embedding
    embedding_dict = get_movie_embedding_dict(unique_movie_ids)
    candidate_movie_ids = unique_movie_ids.copy()
    
	# fine the nearest neighbor movie of the movie id passed on.
    candidate_movie_ids.remove(MOVIE_ID_TO_EXAMINE)
    best_id, min_dist = get_nn_of_movie(MOVIE_ID_TO_EXAMINE, candidate_movie_ids, embedding_dict)
    movieID_name_map = load_movie_id_name_map(LOCAL_MOVIE_LOOKUP_FILE)
    print('The closest movie to {} in the embedding space is {}'.format(movieID_name_map[MOVIE_ID_TO_EXAMINE],
                                                                  movieID_name_map[best_id]))
    candidate_movie_ids.append(MOVIE_ID_TO_EXAMINE)

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }

def get_unique_movie_ids(data_list):
    unique_movie_ids = set()
    for row in data_list:
        unique_movie_ids.add(row['in1'][0])
    return list(unique_movie_ids)

'''
This function calls model endpoint and create movie embedding based on the movies ids 
passed on.
'''
def get_movie_embedding_dict(movie_ids):
    input_instances = list()
    for s_id in movie_ids:
        input_instances.append({"in1": [s_id]})
    data = json.dumps({"instances": input_instances})
    movie_embeddings = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=data)
    
    result = json.loads(movie_embeddings['Body'].read().decode())
    #print(result)
								
    embedding_dict = {}
    for s_id, row in zip(movie_ids, result['predictions']):
        embedding_dict[s_id] = np.array(row['embeddings'])
    return embedding_dict
    

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