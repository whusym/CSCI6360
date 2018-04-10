import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

df = pd.read_csv("./CSCI6360/Project_2/caravan.csv")
data = df.values
#print(data.head())
# print(data.shape)

X = data[:,:-1].astype(float)

# print(X.shape)
# print(X.head())

y = data[:, -1]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# print(y.shape)



def build_model():
    model = Sequential()
    model.add(Dense(85, input_dim=85, activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer = 'normal', activation='sigmoid'))

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=build_model, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_y, cv=kfold)
print("Results: {:.2f}% ({:.2f}%)".format((results.mean()*100, results.std()*100)))
