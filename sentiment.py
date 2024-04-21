import os
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.utils import mediainfo
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import markovify
import pandas as pd

# Sample data
data = {
    'Lyrics': ['lyrics1', 'lyrics2', 'lyrics3'],
    'Emotion': ['happy', 'sad', 'angry']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Write DataFrame to CSV file
df.to_csv('song_lyrics_emotions.csv', index=False)


data = pd.read_csv('song_lyrics_emotions.csv')


def get_audio_info(file_path):
    audio = AudioSegment.from_file(file_path)
    channels = audio.channels
    sample_width = audio.sample_width
    framerate = audio.frame_rate
    num_frames = len(audio.get_array_of_samples())
    duration = len(audio) / 1000.0
    return channels, sample_width, framerate, num_frames, duration, audio

def plot_audio_graph(audio):
    samples = np.array(audio.get_array_of_samples())
    time = np.arange(len(samples)) / audio.frame_rate
    plt.plot(time, samples)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    plt.show()

def classify_genre(audio):
   
    return 'good'

def get_lyrics(artist, song_title):
    url = f"https://www.azlyrics.com/lyrics/{artist.lower().replace(' ', '')}/{song_title.lower().replace(' ', '')}.html"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        lyrics_div = soup.find_all("div", attrs={"class": None, "id": None})
        lyrics = "\n".join([div.get_text() for div in lyrics_div])
        return lyrics
    else:
        return None

def analyze_sentiment(lyrics):
    if lyrics is None:
        print("Lyrics not found.")
        return
    blob = TextBlob(lyrics)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        print("The sentiment of the song is positive.")
    elif sentiment_score < 0:
        print("The sentiment of the song is negative.")
    else:
        print("The sentiment of the song is neutral.")

def train_sentiment_model(data):
    X = data['Lyrics']
    y = data['Emotion']
    emotions_mapping = {'happy': 0, 'sad': 1, 'angry': 2, 'calm': 3}
    y = y.map(emotions_mapping)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    return svm_classifier, vectorizer

def predict_song_sentiment(artist, song_title, classifier, vectorizer):
    lyrics = get_lyrics(artist, song_title)
    if lyrics is None:
        print("Lyrics not found.")
        return
    X_lyrics = vectorizer.transform([lyrics])
    predicted_emotion = classifier.predict(X_lyrics)[0]
    emotions_mapping_rev = {v: k for k, v in emotions_mapping.items()}
    predicted_emotion_label = emotions_mapping_rev[predicted_emotion]
    print("Predicted Emotion:", predicted_emotion_label)

def generate_lyrics(artist, song_title, num_lines):
    url = f"https://www.azlyrics.com/lyrics/{artist.lower().replace(' ', '')}/{song_title.lower().replace(' ', '')}.html"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        lyrics_div = soup.find_all("div", attrs={"class": None, "id": None})
        lyrics_text = "\n".join([div.get_text() for div in lyrics_div])
        lyrics_model = markovify.Text(lyrics_text)
        generated_lyrics = "\n".join([lyrics_model.make_sentence() for _ in range(num_lines)])
        print("Generated Lyrics:")
        print(generated_lyrics)
    else:
        print("Failed to fetch lyrics for generation.")

if __name__ == "__main__":
    file_path = input("Enter the path to the MP3 file: ")

    if not os.path.isfile(file_path):
        print("File not found.")
        exit()

    channels, sample_width, framerate, num_frames, duration, audio = get_audio_info(file_path)
    print("Number of Channels:", channels)
    print("Sample Width:", sample_width)
    print("Frame Rate:", framerate)
    print("Number of Frames:", num_frames)
    print("Duration:", duration, "seconds")

    

    genre = classify_genre(audio)
    print("Genre Classification:", genre)

    artist = input("Enter the artist's name: ")
    song_title = input("Enter the song title: ")

    # data = pd.read_csv('song_lyrics_emotions.csv')  # Assuming you have a CSV file with song lyrics and emotions
    # classifier, vectorizer = train_sentiment_model(data)
    # predict_song_sentiment(artist, song_title, classifier, vectorizer)

    # num_lines = int(input("Enter the number of lines for generated lyrics: "))
    # generate_lyrics(artist, song_title, num_lines)
    plot_audio_graph(audio)
    
