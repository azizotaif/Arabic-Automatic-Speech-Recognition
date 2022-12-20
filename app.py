# --------------------------------------- IMPORTS ----------------------------------------------

import os
import argparse
from flask import Flask, render_template, request
import torch
import librosa
from pydub import AudioSegment
import io
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
import datetime
import ctcdecode
import numpy as np

# ==============================================================================================
# ----------------------------- LOADING THE TRAINED MODELS -------------------------------------

# Loading acoustic model
model_dir = 'models/'
device = torch.device("cpu")
model = Wav2Vec2ForCTC.from_pretrained(model_dir)
processor = Wav2Vec2Processor.from_pretrained(model_dir)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_dir)
model.to(device)
model.eval()

# Loading language model
vocab_dict = tokenizer.get_vocab()
sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
vocab = [x[1].replace("|", " ") if x[1] not in tokenizer.all_special_tokens else "_" for x in sort_vocab]
alpha = 2.5 # LM Weight
beta = 0.0 # LM Usage Reward
word_lm_scorer = ctcdecode.WordKenLMScorer(f'{model_dir}language_model.arpa', alpha, beta)
decoder = ctcdecode.BeamSearchDecoder(
    vocab,
    num_workers=2,
    beam_width=128,
    scorers=[word_lm_scorer],
    cutoff_prob=np.log(0.000001),
    cutoff_top_n=40
)

# ===============================================================================================
# ------------------------------ PREPROCESS AND INFERENCE FUNCTIONS -----------------------------

def preprocess(wav, file_name):
    wav = io.BytesIO(wav)
    if file_name.endswith('.mp3'):
        wav = AudioSegment.from_mp3(wav)
        path = f'./temp/{str(datetime.datetime.now())}.wav'
        wav.export(path, format="wav")
        speech_array, sampling_rate = librosa.load(path, sr=16000)
        os.remove(path)
    elif file_name.endswith('.flac'):
        wav = AudioSegment.from_file(wav)
        path = f'./temp/{str(datetime.datetime.now())}.wav'
        wav.export(path, format="wav")
        speech_array, sampling_rate = librosa.load(path, sr=16000)
        os.remove(path)
    elif file_name.endswith('.wav'):
        path = f'./temp/{str(datetime.datetime.now())}.wav'
        wav = AudioSegment.from_wav(wav)
        wav.export(path, format="wav")
        speech_array, sampling_rate = librosa.load(path, sr=16000)
        os.remove(path)
    
    if sampling_rate != 16000:
        speech_array, s = librosa.load(wav, sr=16000)
    inputs = speech_array

    return inputs


def inference(inputs, model=model):
    with torch.no_grad():
        input_values = torch.tensor(inputs, device=device).unsqueeze(0)
        logits = model(input_values.float()).logits
        inference = decoder.decode_batch(logits.numpy())[0]
    return inference

# ===============================================================================================
# ----------------------------------------- FLASK APP -------------------------------------------


app = Flask(__name__)

# render home page

@ app.route('/')
def home():
    title = 'Arabic ASR'
    return render_template('index.html', title=title)


# ===============================================================================================
# ------------------------------------ RENDER PREDICTION PAGE -----------------------------------

@ app.route('/', methods=['POST'])
def transcribe():
    title = 'Arabic ASR'

    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_name = file.filename
            wav = file.read()

        if wav != None:
            inputs = preprocess(wav, file_name)
            result = inference(inputs)

            return render_template('inference_result.html', inference=result, title=title)
        else:
            return render_template('try_again.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port)