import pyttsx3

print("Starting TTS test...")

engine = pyttsx3.init()

engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)

print("Speaking...")

engine.say("HELLO THIS IS A SENTENCE TEST")
engine.runAndWait()

print("Finished.")

engine.stop()