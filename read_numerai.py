import sys
import numpy as np
import pandas as pd
from pprint import pprint


def read_numerai(path, num_examples, testing_percent, vector_labels):
    # Check testing percentage
    if(testing_percent >= 1 or testing_percent < 0):
        raise ValueError("The testing percentage must be less than 1 and greater than 0: %d" % testing_percent)

    # Adjust path ending if necassary
    if(path[-1] != '/'):
        path = path + '/'

    # Set training and testing file names
    training_filename = "numerai_training_data.csv"
    tournament_filename = "numerai_tournament_data.csv"

    # Read raw data
    raw_training_data = pd.read_csv(path + training_filename)
    tourney_exists = False
    try:
        raw_tournament_data = pd.read_csv(path + tournament_filename)
        tourney_exists = True
    except IOError:
        pass

    # Make sure number of samples does not exceed count of traning and testing data
    max_train_count = int(raw_training_data.shape[0] * (1 - testing_percent))
    max_test_count = int(raw_training_data.shape[0] - max_train_count)
    if(num_examples > max_train_count + max_test_count):
        raise ValueError("The number of samples cannot exceed the sum of the training and testing data: %d" % num_examples)

    # Set the number of training and testing examples
    train_count = int(num_examples * (1 - testing_percent))
    test_count = int(num_examples - train_count)

    # Extract training, testing, and tournament data
    train_features = raw_training_data[:train_count].as_matrix(columns=raw_training_data.columns[:-1])
    if(test_count != 0):
        test_features = raw_training_data[-test_count:].as_matrix(columns=raw_training_data.columns[:-1])
    else:
        test_features = []
    tourney_features = None
    tourney_ids = None
    if(tourney_exists):
        tourney_features = raw_tournament_data.as_matrix(columns=raw_tournament_data.columns[1:])
        tourney_ids = raw_tournament_data['t_id'].values

    # Setup labels as single- or multi-dimensional structures
    if(vector_labels):
        train_labels = np.ndarray(shape=(train_count, 2))
        for i in range(train_count):
            if(raw_training_data['target'].values[i] == 0):
                train_labels[i,:] = [1, 0]
            else:
                train_labels[i,:] = [0, 1]

        if(test_count != 0):
            test_labels = np.ndarray(shape=(test_count, 2))
            for i in range(test_count):
                if(raw_training_data['target'].values[-test_count + i] == 0):
                    test_labels[i,:] = [1, 0]
                else:
                    test_labels[i,:] = [0, 1]
        else:
            test_labels = []
    else:
        train_labels = raw_training_data['target'][:train_count-1].values
        if(test_count != 0):
            test_labels = raw_training_data['target'][-test_count:].values
        else:
            test_labels = []

    return train_features, test_features, tourney_features, train_labels, test_labels, tourney_ids

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print "Please provide valid input parameters ([path to numerai files] [number of exaples] [percentage of training for testing] [vector labels?])"
        exit(1)
    path = sys.argv[1]
    num_examples = int(sys.argv[2])
    test_perc = float(sys.argv[3])
    vector_labels = int(sys.argv[4])

    train_features, test_features, tourney_features, train_labels, test_labels, tourney_ids = read_numerai(path, num_examples, test_perc, vector_labels)

    pprint(train_features)
    pprint(test_features)
    pprint(tourney_features)
    pprint(train_labels)
    pprint(test_labels)
    pprint(tourney_ids)
