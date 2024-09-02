import speech_recognition as sr


r = sr.Recognizer()

mic = sr.Microphone()

print('Say something!')
with mic as source:
    audio = r.listen(source)

result = r.recognize_whisper(audio)
print(result)