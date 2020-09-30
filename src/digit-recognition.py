import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


def read(file_path, length=35):
    files = os.listdir(file_path)
    feature = []
    labels = []

    for file in files:
        y, sr = librosa.load(file_path + file)
        mfcc_features = librosa.feature.mfcc(y)
        if mfcc_features.shape[1] > length:
            mfcc_features = mfcc_features[:, 0:length]
        else:
            mfcc_features = np.pad(mfcc_features, ((0, 0), (0, length - mfcc_features.shape[1])), mode='constant',
                                   constant_values=0)
        feature.append(np.array(mfcc_features))

        label = np.eye(10)[int(file[0])]
        labels.append(label)

    return np.asarray(feature), np.array(labels)


ft_batch, label_batch = read('../data/recordings/')
print("read finish")
x_train, x_test, y_train, y_test = train_test_split(ft_batch, label_batch, test_size=0.2, random_state=42)

model = Sequential()
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(128, dropout=0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('compilation done...')
history = model.fit(x_train, y_train,
                    epochs=1000,
                    batch_size=128)

print('training completed')
model.save("../model/voice2digit.h5")
score = model.evaluate(x_test, y_test)
print(score)
