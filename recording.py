import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from translatepy.translators.google import GoogleTranslate
import speech_recognition as sr

# Function to extract audio features for speaker recognition
# Uses the MFCC (Mel Frequency Cepstral Coefficients) technique for feature extraction
def extract_features(audio_file, n_mfcc=40):
    try:
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=None)
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Return the mean of MFCCs along the time axis
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        # Handle errors during audio file processing
        print(f"Error processing {audio_file}: {e}")
        return None

# Function to recognize speech from an audio file using Google's Speech Recognition API
def recognize_speech(audio_file):
    recognizer = sr.Recognizer()
    try:
        # Load the audio file
        with sr.AudioFile(audio_file) as source:
            # Record audio data from the file
            audio_data = recognizer.record(source)
            # Recognize and return text from the audio
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        # Handle case where speech is not understandable
        return "Could not understand the audio."
    except sr.RequestError:
        # Handle errors with the Speech Recognition API
        return "Speech Recognition API is not available."
    except Exception as e:
        # Handle other exceptions
        return f"Error during speech recognition: {e}"

# Function to translate text into a specified language using GoogleTranslate
def translate_text(text, target_language="ur"):
    try:
        translator = GoogleTranslate()
        # Translate the input text into the target language
        result = translator.translate(text, target_language)
        return result
    except Exception as e:
        # Handle translation errors
        return f"Translation failed: {e}"

# Function to train the speaker recognition model
def train_speaker_model(audio_files, labels):
    features = []  # To store extracted audio features
    valid_labels = []  # To store corresponding labels for valid audio files
    for f, label in zip(audio_files, labels):
        # Extract features for each audio file
        feature = extract_features(f)
        if feature is not None:
            features.append(feature)
            valid_labels.append(label)

    # Ensure that valid audio features were extracted
    if not features:
        raise ValueError("No valid audio features were extracted. Please check your audio files.")

    # Convert features and labels into numpy arrays
    features = np.array(features)
    label_encoder = LabelEncoder()
    # Encode string labels into numerical format
    encoded_labels = label_encoder.fit_transform(valid_labels)

    # Train an SVM (Support Vector Machine) model with a linear kernel
    model = SVC(kernel="linear", probability=True)
    model.fit(features, encoded_labels)
    return model, label_encoder

# Main program
def main():
    print("Welcome to the Speaker Recognition and Speech-to-Text Program!")
    users_data = {}  # Dictionary to store user names and their associated audio files

    # Step 1: Training Phase
    print("Training Phase: Add users and their audio samples.")
    while True:
        # Input the user's name
        user_name = input("Enter the user's name: ").strip()
        if not user_name:
            print("User name cannot be empty.")
            continue

        # Input audio files for the user
        print(f"Enter audio files for {user_name}. Type 'done' when you finish.")
        audio_files = []
        while True:
            audio_file = input("Enter the path to an audio file (WAV format) or 'done' to finish: ").strip()
            if audio_file.lower() == 'done':
                break
            if not os.path.exists(audio_file) or not audio_file.endswith(".wav"):
                print(f"Invalid file path: {audio_file}. Please enter a valid .wav file.")
                continue
            audio_files.append(audio_file)

        # Store data for the user
        if user_name in users_data:
            users_data[user_name].extend(audio_files)
        else:
            users_data[user_name] = audio_files

        # Ask if the user wants to add another user
        another_user = input("Do you want to add another user? (yes/no): ").strip().lower()
        if another_user != 'yes':
            break

    # Train speaker recognition model
    all_audio_files = []
    all_labels = []
    for user_name, audio_files in users_data.items():
        all_audio_files.extend(audio_files)
        all_labels.extend([user_name] * len(audio_files))

    print("Training the speaker recognition model...")
    try:
        speaker_model, label_encoder = train_speaker_model(all_audio_files, all_labels)
        print("Training completed successfully!")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Step 2: Recognition Phase
    print("\nRecognition Phase: Test the trained model.")
    target_language = input("Enter the target language code for translation (e.g., 'ur' for Urdu, 'ta' for Tamil): ").strip()

    while True:
        test_file = input("\nEnter the path to a test audio file (WAV format): ").strip()
        if not os.path.exists(test_file) or not test_file.endswith(".wav"):
            print(f"Invalid file path: {test_file}. Please enter a valid .wav file.")
            continue

        # Recognize speech from the audio file
        recognized_text = recognize_speech(test_file)
        print(f"Recognized Text: {recognized_text}")

        # Identify the speaker
        test_features = extract_features(test_file)
        if test_features is not None:
            speaker_label = label_encoder.inverse_transform(speaker_model.predict([test_features]))[0]
            print(f"Identified Speaker: {speaker_label}")
        else:
            print("Failed to extract features from the test audio.")

        # Translate the recognized text into the target language
        translated_text = translate_text(recognized_text, target_language)
        print(f"Translated Text ({target_language}): {translated_text}")

        # Ask if the user wants to test another file
        another_test = input("Do you want to test another file? (yes/no): ").strip().lower()
        if another_test != 'yes':
            break

    print("Thank you for using the program!")

if __name__ == "__main__":
    main()
