from Nested import Experiments_Nested
import pandas as pd
import sys

#arg 1 = model type i.e. linear, rf etc
#arg 2 = experiment 1 or 2
#arg 3 = opposition bool
#arg 4 = input type
#arg 5 = x data
#arg 6 = y data
#arg 7 = ngram (F) or we engine (T)

exp = Experiments_Nested([f'{sys.argv[1]}'], experiment=f'{sys.argv[2]}', opposition=eval(sys.argv[3]), input_representation=f'{sys.argv[4]}')

X_train = pd.read_pickle(f'{sys.argv[5]}')
y_train = pd.read_pickle(f'{sys.argv[6]}')

y_train = y_train[0].to_numpy()

if eval(sys.argv[7]) is True:
    results = exp.training_loop_we(X_train,y_train)
else:
    results = exp.training_loop(X_train, y_train)















