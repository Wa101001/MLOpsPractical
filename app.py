from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import pickle


def custom_ngram(x):
    return (x[-i-1:] for i in range(0, min(3, len(x))))
def load_pickled_object(file_path):
    with open(file_path, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object

app = Flask(__name__)

# Load the saved objects
loaded_vect = load_pickled_object('vect.pickle')
loaded_feature_names = load_pickled_object('feature_names.pickle')
loaded_model = load_pickled_object('linear_svm_model.pickle')

def preprocess_input(user_input):
    user_input_transformed = loaded_vect.transform([user_input])
    grams = pd.DataFrame(user_input_transformed.todense(), columns=loaded_feature_names)
    return grams

def make_prediction(grams):
    prediction = loaded_model.predict(grams)
    return prediction[0]  
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.form['user_input']
        grams = preprocess_input(user_input)  # Get grams DataFrame
        prediction = make_prediction(grams)    # Pass grams to make_prediction
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return render_template('index.html', error=str(e))
if __name__ == '__main__':
    app.run(debug=True)