import speech_recognition as sr
def recognize_speech():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Use the microphone as the source of input
    with sr.Microphone() as source:
        
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)
        
        # Listen to the speech
        audio = recognizer.listen(source)

        try:
            # Recognize the speech using Google Web Speech API
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            text="Sorry, I could not understand the audio."
            return text
        except sr.RequestError:
            text="Sorry, my speech recognition service is down."
            return text