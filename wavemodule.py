# import matplotlib.pyplot as plt
# import numpy as np
# from pydub import AudioSegment
# from pydub.utils import mediainfo
# import librosa
# import librosa.display
# from sklearn.ensemble import RandomForestClassifier

# def extract_audio_info(file_path):
#     audio = AudioSegment.from_mp3(file_path)
#     audio_info = mediainfo(file_path)
#     num_channels = audio.channels
#     sample_width = audio.sample_width
#     frame_rate = audio.frame_rate
#     num_frames = len(audio)
#     duration = len(audio) / 1000.0
#     return audio, audio_info, num_channels, sample_width, frame_rate, num_frames, duration

# def plot_audio_waveform(audio, sample_rate):
#     audio_data = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0  # Convert to floating-point and normalize
#     plt.figure(figsize=(10, 4))
#     librosa.display.waveshow(audio_data, sr=sample_rate)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#     plt.title('Audio Waveform')
#     plt.show()


# def genre_classification(audio_info):
#     X = [[audio_info['channels'], audio_info['sample_width'], audio_info['frame_rate'], audio_info['duration']]]
#     y = ['Sad', 'Angry', 'Good']  
#     clf = RandomForestClassifier()
#     clf.fit(X, y)
#     prediction = clf.predict(X)[0]
#     return prediction
# def main():
#     file_path = input("Enter the path to the MP3 file: ")
#     audio, audio_info, num_channels, sample_width, frame_rate, num_frames, duration = extract_audio_info(file_path)

#     print("Audio Information:")
#     print("Number of Channels:", num_channels)
#     print("Sample Width:", sample_width)
#     print("Frame Rate:", frame_rate)
#     print("Number of Frames:", num_frames)
#     print("Duration:", duration, "seconds")

#     # plot_audio_waveform(audio, frame_rate)  # Call plot_audio_waveform function here
#     import matplotlib.pyplot as plt

# def plot_audio_waveform(audio, sample_rate):
#     audio_data = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0  # Convert to floating-point and normalize
#     time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
#     plt.figure(figsize=(10, 4))
#     plt.plot(time, audio_data)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#     plt.title('Audio Waveform')
#     plt.pause(0.01)



#     genre = genre_classification(audio_info)
#     print("Genre Classification:", genre)

# if __name__ == "__main__":
#     main()


# def main():
#     file_path = input("Enter the path to the MP3 file: ")
#     audio, audio_info, num_channels, sample_width, frame_rate, num_frames, duration = extract_audio_info(file_path)

#     print("Audio Information:")
#     print("Number of Channels:", num_channels)
#     print("Sample Width:", sample_width)
#     print("Frame Rate:", frame_rate)
#     print("Number of Frames:", num_frames)
#     print("Duration:", duration, "seconds")

#     # plot_audio_waveform(audio, frame_rate)
# #    def plot_audio_waveform(audio, sample_rate):
# def plot_audio_waveform(audio, sample_rate):
#     # Function body goes here
#     # Ensure proper indentation
#     pass  # Placeholder, replace with actual code

#     audio_data = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0  # Convert to floating-point and normalize
#     time = np.arange(0, len(audio_data)) / sample_rate
#     plt.figure(figsize=(10, 4))
#     plt.plot(time, audio_data)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#     plt.title('Audio Waveform')
#     plt.show()


    
#     genre = genre_classification(audio_info)
#     print("Genre Classification:", genre)

# if __name__ == "__main__":
#     main()

