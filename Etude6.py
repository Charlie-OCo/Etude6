import time
import numpy as np
import scipy.signal
import librosa
import librosa.display
import matplotlib.pyplot as plt
 
# Load audio file
filename = 'song1.wav'  # Replace with your file
y, sr = librosa.load(filename, sr=48000)  # y is the waveform, sr is the sample rate
 
songLength = 4
pieceLength = 0.1
for i in range(int(songLength / pieceLength)):
    # Select a short segment (e.g., first 0.5 seconds)
    beginning = i * pieceLength 
    end = beginning + pieceLength 
    piece = y[int(sr * beginning):int(sr * end)]

    # Apply a window and zero-padding
    windowed = piece * np.hanning(len(piece))
    n_fft = 4096  # Zero padding for better resolution
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, 1/sr)
    # Plot spectrum
    plt.plot(frequencies, spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT Spectrum')
    plt.ylim(0, 8)  # Use max magnitude from first pass
    plt.pause(pieceLength)
    plt.clf()  # Clear the plot for the next iteration