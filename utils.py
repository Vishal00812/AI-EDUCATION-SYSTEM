import speech_recognition as sr
import pyttsx3
import keyboard
import threading
import time
import streamlit as st
def speak(text):
    engine = pyttsx3.init()
    
    # Set properties for speech (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # You can change the voice index (0 for male, 1 for female)
    
    # Run the speech engine in a separate thread
    def run_speech():
        engine.say(text)
        engine.runAndWait()

    # Start speaking in a separate thread to allow real-time stopping
    speech_thread = threading.Thread(target=run_speech)
    speech_thread.start()

    # Continuously check if "Enter" is pressed
    while speech_thread.is_alive():
        if keyboard.is_pressed('enter'):
            engine.stop()
            break
        time.sleep(0.1)  # Small delay to reduce CPU usage


        

def takeCommand():
    #It takes microphone input from the user and returns string output

    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        query = r.recognize_google(audio, language='en-in')

    except Exception as e:

        return "None"
    return query