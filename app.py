from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load pre-trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    # Vectorize input message
    message_vect = vectorizer.transform([message])
    
    # Predict using the model
    prediction = model.predict(message_vect)
    
    # Return result
    result = 'spam' if prediction == 1 else 'not_spam'
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
