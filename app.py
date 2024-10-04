from flask import Flask
from CreateAllModels import create_models

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/create_models')
def create_models_route():
    create_models()
    return "Models created successfully!"

if __name__ == '__main__':
    app.run(debug=True)

    