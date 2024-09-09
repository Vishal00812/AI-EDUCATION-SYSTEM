from flask import Flask, render_template, redirect
import os

app = Flask(__name__)

@app.route('/')
def home():
    # Render the HTML page with the "Run App" button
    return render_template('index.html')

@app.route('/run-app')
def run_app():
    # This runs the Streamlit app (app.py) when the button is clicked
    os.system("streamlit run app.py")  # Make sure 'app.py' is in the same directory
    return redirect('/')  # Redirect back to the home page after starting Streamlit

if __name__ == '__main__':
    # Start the Flask app
    app.run(debug=True)
