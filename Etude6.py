import time
import numpy as np
import scipy.signal
import librosa
import librosa.display
import matplotlib.pyplot as plt
 
# Load audio file
filename = 'song9.wav'  # Replace with your file
y, sr = librosa.load(filename, sr=48000)  # y is the waveform, sr is the sample rate

rawNotes = []

songLength = 4 
pieceLength = 0.2
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

    maxFound =0
    bestIndex = 0
    for j in range(0, 2000): 
        if spectrum[j] > maxFound:
            maxFound = spectrum[j]
            bestIndex = j
    
    highestFreq = frequencies[bestIndex]
    
    def freq_to_note(freq):
        A4 = 440.0
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        if freq <= 0:
            return "No note"
        
        # Calculate number of semitones away from A4
        n = int(round(12 * np.log2(freq / A4)))
        
        # Find the note name and octave
        note_index = (n + 9) % 12  # A is index 9 in the list
        octave = 4 + ((n + 9) // 12)
        
        return f"{notes[note_index]}{octave}"

    note = freq_to_note(highestFreq)
    rawNotes.append(note)
    plt.plot(frequencies, spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT Spectrum')
    plt.ylim(0, 11)  # Use max magnitude from first pass
    plt.xlim(0, 3000)
    plt.text(100, 2, note, fontsize=30, color='red')  # X=100 Hz, Y=90% of max

    plt.pause(pieceLength)
    plt.clf()  # Clear the plot for the next iteration

    
notes = [rawNotes[0]]
for note in rawNotes:
    if note != notes[-1]:
        notes.append(note)

for note in notes:
    if note == "No note":
        notes.remove(note)
    
for note in notes:
    print(note, "-", end = "") 