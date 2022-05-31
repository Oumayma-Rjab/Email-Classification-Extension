from flask import Flask,render_template,url_for,request,json,jsonify
import os
from grpc import protos_and_services
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
print(os.getcwd())
loaded_model = load_model("/home/change-it/Documents/RT3INSAT/ppp/project/api/my_model.h5")
tokenizer = None
with open('/home/change-it/Documents/RT3INSAT/ppp/project/api/tokenizer.pkl', 'rb') as f:
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
        my_prediction = loaded_model.predict(testdata)[0]
        idx=np.argmax(my_prediction)
        categories = ["daily", 'demand documents', 'contacts', 'deleted items',
       'discussion threads', 'inbox', 'notes', 'work items', 'sent',
       'straw', '2000 conference', 'active international', 'avaya', 'bmc',
       'bridge', 'bristol babcock', 'colleen koenig', 'compaq',
       'computer associates', 'continental airlines']
        print(idx)
        res['prediction'] = categories[idx]
    return jsonify(res)

    
if __name__ == "__main__":
    port = os.environ.get("PORT",5800)
    app.run(debug=False,host="0.0.0.0",port=prot)