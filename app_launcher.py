from flask import Flask, render_template, redirect
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run-app')
def run_app():
    # Command to run the Streamlit app (this assumes `app.py` is in the same directory)
    os.system("streamlit run app.py")
    return redirect('/')  # After starting, redirect to the home page

if __name__ == '__main__':
    app.run(debug=True)
