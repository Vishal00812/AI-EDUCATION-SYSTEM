from flask import Flask
import subprocess
import webbrowser
import time
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def start_server():
    # Start the Node.js server
    subprocess.Popen(['node', 'index.js'], cwd=os.path.join(os.getcwd(), 'server'))

def start_client():
    # Start the client
    os.system("cd client && npm run dev")
def start_streamlit():
    # Start the Streamlit app
    subprocess.Popen(['streamlit', 'run', 'app.py'], cwd=os.path.join(os.getcwd(), 'chatbot'))

@app.route('/')
def index():
    return "Flask app is running!"

if __name__ == '__main__':
    # Start the server, client, and Streamlit in parallel
    start_server()

    # Allow some time for the server to start
    time.sleep(5)

    start_client()
    #start_streamlit()

    # Open the website in a web browser
    time.sleep(10)  # Wait for the client to fully start
    webbrowser.open('http://localhost:5173')  # Adjust the port if needed

    # Run Flask app
    app.run(port=5000, debug=True)
