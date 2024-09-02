import speech_recognition as sr

r = sr.Recognizer()
mic = sr.Microphone()
def recognize_speech_from_mic(recognizer, microphone):
    with mic as source:
        audio = r.listen(source)
        response = {
            "success": True,
            "error": None,
            "transcription": None
        }
        try:
            response["transcription"] = recognizer.recognize_google(audio)
        except sr.RequestError:
            # API was unreachable or unresponsive
            response["success"] = False
            response["error"] = "API unavailable"
        except sr.UnknownValueError:
            # speech was unintelligible
            response["error"] = "Unable to recognize speech"
        return response

print("Speak now:")
guess = recognize_speech_from_mic(r, mic)
if guess["transcription"]:
    print(guess["transcription"])



