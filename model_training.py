from tensorflow.keras.layers import LSTM, Dense, Dropout ,LeakyReLU ,Flatten,GRU
import keras
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from datetime import datetime 
path = 'MP_Data'
actions = os.listdir(path)
x_train, y_train = [], []
xtmp, ytmp = [], []
for files in actions:
    try:
        while True:
            xtmp.append(np.load(os.path.join(path, files, str.endswith('.npy'))))
            ytmp.append(files)
    except:
        pass
    x_train.append(xtmp)
    y_train.append(ytmp)
    xtmp, ytmp = [], []
y_train = to_categorical(y_train)
Xt, Yt, xt, yt = train_test_split(x_train, y_train, test_size=0.8)


model = keras.Sequential(name='model')
model.add(LSTM(256, return_sequences=True, activation="tanh", input_shape=(30,1662)))
model.add(Dropout(0.1))
model.add(LSTM(128, return_sequences=True, activation="tanh"))
model.add(Dropout(0.1))
model.add(LSTM(64, return_sequences=True, activation="tanh"))
model.add(Dropout(0.1))
model.add(LSTM(32, return_sequences=False, activation="tanh"))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['acc']
)
model.fit(Xt, Yt, epochs = 1000, batch_size = 128)
os.mkdir('model')
model.save(f'model/model{str(datetime.today())}.h5')
