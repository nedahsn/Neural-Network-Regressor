# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:34:04 2021

@author: Neda
"""
# Build a NN Regressor model to predict popularity in Spotify data 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit


from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import callbacks


spotify = pd.read_csv('E:/OneDrive/Neda/Machine learning/Kaggle/Spotify/Data/spotify.csv')

# Due to small number of missing values, we just drop rows with missing values
X = spotify.copy().dropna()

# Creat target
y = X.pop('track_popularity')

artists = X['track_artist']

# Numerical features
features_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']

# Categorical features
features_cat = ['playlist_genre']


# WE didn't consider the following features:
# "track_id", "track_album_id", "playlist_id", because they do not affect popularity
# "track_name", "track_album_name", "playlist_name", "track_album_release_date", because they do not seem to affect popularity   
# "playlist_subgenre", because it has a high number of cardinality of 24 



# Making preprocessor for Standardization of numerical features
# and OneHotEncoding of categorical features
preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat),
)


# We'll do a "grouped" split to keep all of an artist's songs in one
# split or the other. This is to help prevent signal leakage.
# GroupShuffleSplit provides randomized train/test indices to split data 
# according to a third-party provided group

def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

X_train, X_valid, y_train, y_valid = group_split(X, y, artists)


# Fit and transform based on the preprocessor
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

# popularity is on a scale 0-100, so this rescales to 0-1.
y_train = y_train / 100 
y_valid = y_valid / 100

input_shape = [X_train.shape[1]]
print("Input shape: {}".format(input_shape))

# We now build a NN Regressor model to predict popularity in Spotify data


## We first start with a single-layer NN - Underfitting
model = keras.Sequential([
    layers.Dense(1, input_shape=input_shape),
])

# a regressor with loss "mae"
model.compile(
    optimizer='adam',
    loss='mae',
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    verbose=0, # suppress output since we'll plot the curves
)

history_df = pd.DataFrame(history.history)
history_df.loc[10:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
plt.title("Underfitting")

## We next build a multi-layer NN - Overfitting
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mae',
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
plt.title("Overfitting")

## We now define Early Stopping Callback to prevent over fitting
#   an early stopping callback that waits 5 epochs (patience')
#   for a change in validation loss of at least 0.001 (min_delta)
#   and keeps the weights with the best loss (restore_best_weights):
    
early_stopping = callbacks.EarlyStopping(min_delta=0.001, patience=5,
                                        restore_best_weights=True)


model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),    
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mae',
)

# a NN with Early Stopping Callback
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    callbacks=[early_stopping]
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
plt.title("Early Stopping Callback to prevent overfitting")

# Add Dropout to prevent over fitting
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dropout(rate=0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(rate=0.3),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mae',
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    verbose=0,
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
plt.title("Add Dropout to prevent overfitting")









