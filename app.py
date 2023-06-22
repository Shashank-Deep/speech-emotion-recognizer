import numpy as np
import librosa
import tensorflow as tf
import io
# import sounddevice as sd
# import soundfile as sf
from flask import Flask, render_template, request

# Load the model
# replace 'my_model.h5' with your own saved model
model = tf.keras.models.load_model('my_model.h5')

# Define emotions
emotions = {
    0: 'angry',
    1: 'fear',
    2: 'happy',
    3: 'neutral',
    4: 'sad',
    5: 'surprise'
}

# Define function to extract MFCC features
def extract_mfcc(audio_bytes, sr):
    y, sr = librosa.load(io.BytesIO(audio_bytes), duration=3, offset=0.5, sr=sr)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'audio' in request.files:
            audio_file = request.files['audio']
            audio_bytes = audio_file.read()

            # Extract MFCC features
            sr = None
            mfcc = extract_mfcc(audio_bytes, sr)
            mfcc = np.reshape(mfcc, newshape=(1, 40))

            # Make prediction using the model
            predictions = model.predict(mfcc)
            emotion = emotions[np.argmax(predictions[0])]

            return render_template('index.html', emotion=emotion)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
