from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import dill

app = Flask(__name__)


# Load feature_names from a file
with open('feature_names.pickle', 'rb') as feature_names_file:
    loaded_feature_names = dill.load(feature_names_file)

# Load vect from a file
with open('vect.pickle', 'rb') as vect_file:
    loaded_vect = dill.load(vect_file)

# Load model from a file
with open('linear_svm_model.pickle', 'rb') as model_file:
    loaded_model = dill.load(model_file)


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
