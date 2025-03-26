# Spam Message Detection

## Overview
This project is a **Spam Message Detection System** built using **Python** and **Flask**. It classifies messages as **spam** or **not spam** using a **Multinomial Naive Bayes (MultinomialNB)** model.

## Features
- Classifies messages as **spam** or **ham (not spam)**.
- Built using **Flask** for the web interface.
- Uses **Natural Language Processing (NLP)** techniques.
- Model trained on **SMS Spam Collection Dataset**.
- Real-time message classification.

## Technologies Used
- **Python**
- **Flask** (Backend Framework)
- **Scikit-learn** (ML Model)
- **Pandas & Numpy** (Data Processing)
- **HTML, CSS, JavaScript** (Frontend)

## Installation
### Prerequisites
Make sure you have **Python 3.7+** installed on your system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/spam-detection.git
cd spam-detection
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Project
```bash
python app.py
```


## Usage
1. Open the web application in your browser.
2. Enter a message in the input field.
3. Click the **Predict** button to classify the message as **spam** or **not spam**.
4. The background color will change to **red for spam** and **green for non-spam**.

## Project Structure
```
├── static/                 # CSS, JavaScript files
├── templates/              # HTML files
├── model.pkl               # Trained ML model
├── vectorizer.pkl          # Text vectorization model
├── app.py                  # Flask application
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
```

## Dataset
The model is trained on the **SMS Spam Collection Dataset**, which contains **5,574 messages** labeled as spam or ham.

## Future Enhancements
- Deploy on **Render/Heroku** for online access.
- Improve accuracy with deep learning models.
- Support multi-language spam detection.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

## License
This project is licensed under the **MIT License**.
