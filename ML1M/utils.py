import pandas as pd 
import numpy as np
import random as rd

# NOT USED FUNCTION
def get_item_movie():
    """List of the name of the movies registered by their item number"""
    list_item_movie=[]
    with open(r"data/movies.dat") as datFile:
        for data in datFile :
            item_movie = data.split()[0]
            clean_item_movie = item_movie.replace('::', ' ')
            item, movie = clean_item_movie.split()
            print(item)
            print(movie)
            list_item_movie.append({item:movie})
            print(list_item_movie)        
    return data
      

def load_data(filename="ratings.dat"):
    clean_data=[]
    with open("ML1M/data/%s"%filename, "r") as dataFile:
       for data in dataFile :
            data = (data.replace("::",',')).strip('\n')
            clean_data.append(data.split(','))

    np_data=np.array([np.array(xi) for xi in clean_data])
    df = pd.DataFrame(np_data, columns=["user","movie","rating","timestamp"])
    return df



def write_small_dataset(N=50, M=200, filename="small_ratings.dat"):
    "Write the first N items of the rating dataset for the M first users "
    df = load_data()
    all_users = df['user'].unique().tolist()
    kept_users = all_users[0:N]

    with open("ML1M/data/%s"%filename,"w") as smallfile:
        for user in kept_users:
            data = df.loc[df.user==user]
            startindex = data.index[0]
            endindex = startindex + M
            data_lines = data.loc[startindex:endindex]
            for ind in data_lines.index:
                smallfile.write('::'.join([str(element) for element in data_lines.loc[ind]]))
                smallfile.write('\n')


def write_ratings(df, NUM_NEG = 99, test_filename = "ml-1m.test.rating",
        train_filename = "ml-1m.train.rating", neg_filename = "ml-1m.test.negative"):
    """Get the data, write one item for the test and the rest for the training"""

    #get the list of all the movies interacted with in this dataset
    movies = df['movie'].unique().tolist()
    # get the list of the users 
    users = df['user'].unique().tolist()

    with open("data/%s"%test_filename, "w") as test_file:
        with open("data/%s"%train_filename, "w") as train_file:
            with open("data/%s"%neg_filename, "w") as neg_file :
                for user in users:
                    data = df.loc[df.user==user]
                    startindex = data.index[0]
                    endindex = startindex + data.shape[0]-1
                    user_nb = int(user) - 1 

                    # write the last rating for the test
                    test_line = data.loc[endindex]
                    test_line["user"] = user_nb
                    test_file.write('\t'.join([str(element) for element in test_line]))
                    test_file.write('\n')

                    # write the other ratings for the training
                    train_lines = data.loc[startindex:(endindex-1)]
                    for ind in train_lines.index:
                        train_line = train_lines.loc[ind]
                        train_line["user"] = user_nb
                        train_file.write('\t'.join([str(element) for element in train_line]))
                        train_file.write('\n')
                    
                    # write the negatives instances 
                    # get the list of movies interact with by the user
                    user_movies = data['movie'].unique().tolist()

                    # create the list of negatives instances allowed 
                    neg_inst = []
                    for movie in movies:
                        if movie not in user_movies:
                            neg_inst.append(movie)
                    
                    # randomly select N instances of negatives instances
                    neg_selection = rd.choices(neg_inst, k=NUM_NEG)

                    neg_file.write('('+str(test_line[0])+','+str(test_line[1])+')\t')
                    neg_file.write('\t'.join([str(element) for element in neg_selection]))
                    neg_file.write('\n')



def launch():
    print("1- Write full dataset \n2- Write default small dataset \n3- Write new small dataset")
    mode=0
    while mode not in [1,2,3]:
        mode = int(input("Mode :"))

    if mode==1:
        df = load_data()
        write_ratings(df)
    if mode == 2:
        print("Number user : 50\n Number movie per user : 200 \n Number neg instances : 20")
        test_filename = "small_ml-1m.test.rating"
        train_filename = "small_ml-1m.train.rating"
        neg_filename = "small_ml-1m.test.negative"
        write_small_dataset()
        df = load_data("small_ratings.dat")
        NUM_NEG=99
        write_ratings(df, NUM_NEG, test_filename, train_filename, neg_filename)
    if mode == 3:
        test_filename = "new_small_ml-1m.test.rating"
        train_filename = "new_small_ml-1m.train.rating"
        neg_filename = "new_small_ml-1m.test.negative"
        N = int(input("Number of users to keep :"))
        M = int(input("Number of movies per user to keep :"))
        NUM_NEG = 99
        filename = "new_small_ratings.dat"
        write_small_dataset(N, M, filename)
        df = load_data(filename)
        write_ratings(df, NUM_NEG, test_filename, train_filename, neg_filename)

launch()
