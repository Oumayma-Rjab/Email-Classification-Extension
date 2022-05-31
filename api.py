from flask import Flask,render_template,url_for,request,json,jsonify
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import sys
import pickle

app = Flask(__name__)

a =  "Hello Flask"

max_vocab = 300
max_len = 150

loaded_model = load_model("my_model.h5")
tokenizer = None
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
print("tokenizer loaded")


@app.route("/detect", methods=["POST"])
def detect():
    my_prediction = ""
    req_data = request.get_json()
    data = json.loads(request.data.decode('UTF-8'))
    message = data['data']
    res = {}
    if(message):
        testtext = []
        testtext.append(message)
        testmsg = np.asarray(testtext)
        testseq = tokenizer.texts_to_sequences(testmsg)
        testdata = pad_sequences(testseq, maxlen=max_len)
        my_prediction = loaded_model.predict_classes(testdata)[0][0]
        res['prediction'] = str(my_prediction)
    return jsonify(res)

    
if __name__ == "__main__":
    app.run()