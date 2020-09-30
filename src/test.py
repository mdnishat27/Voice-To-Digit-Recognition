import tensorflow.keras as keras
import numpy as np
import librosa

length = 35
model = keras.models.load_model("../model/voice2digit94.h5", compile=False)
raw_w, sampling_rate = librosa.load('../data/recordings/1_theo_49.wav', mono=True)

mfcc_features = librosa.feature.mfcc(raw_w)

if mfcc_features.shape[1] > length:
    mfcc_features = mfcc_features[:, 0:length]
else:
    mfcc_features = np.pad(mfcc_features, ((0, 0), (0, length - mfcc_features.shape[1])), mode='constant',
                           constant_values=0)

mfcc_features = mfcc_features.reshape((1, mfcc_features.shape[0], mfcc_features.shape[1]))

prediction_digit = model.predict(mfcc_features)

print("Predicted Digit: ", np.argmax(prediction_digit))
