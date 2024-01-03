import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split

def get_data(test_size=0.2, val_size=0.2, seed=None, print_stats=True):
    
    PATH = "./semeval-2017-tweets_Subtask-A/downloaded/"
    FILES = [PATH+f for f in listdir(PATH) if isfile(join(PATH, f))]
    DFS = pd.concat([pd.read_csv(file,sep="\t",names=['ID',"label",'text']) for file in FILES[:9]+FILES[10:]])
    # There is something wrong with the file twitter-2016test-A.tsv
    # We do not know why or how, but importing it separately and concatenating seems to work
    file10 = pd.read_csv(FILES[10],sep="\t",names=['ID',"label",'text'])
    DFS = pd.concat([DFS, file10])
    
    y = DFS['label']
    X = DFS.drop(columns=['label'])
    

    if print_stats:
        print(f"Length all instances: {len(X)}")
        for unique in set(y):
            print(f"{unique}: {len(y[(y == unique)])}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    split_id = int(len(X_train)*val_size)
    X_train, y_train, X_val, y_val = X_train[split_id:], y_train[split_id:], X_train[:split_id], y_train[:split_id]
    
    if print_stats:
        for _n, _x, _y in [('TRAIN',X_train, y_train), ('VAL', X_val, y_val), ('TEST', X_test, y_test)]:
            print(f"Length all instances {_n}: {len(_x)}")
            for unique in set(_y):
                print(f"{unique}: {len(_y[(_y == unique)])}")
    
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
# get_data()

# Something with average text length could be added as statistic
# But text has to be tokenized for that
