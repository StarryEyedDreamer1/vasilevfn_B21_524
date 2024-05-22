import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import itertools

DPI = 1000

def draw_spectrogram(audio_samples, sample_rate, filename):
    frequencies, time, spectrogram = signal.spectrogram(audio_samples, sample_rate, scaling='spectrum', window=('hann'))

    spectrogram = np.log10(spectrogram + 1)
    plt.pcolormesh(time, frequencies, spectrogram, shading='gouraud', vmin=spectrogram.min(), vmax=spectrogram.max())
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.savefig(filename)
    return frequencies, time, spectrogram

def apply_butterworth_filter(sample_rate, audio_data, output_directory):
    b, a = signal.butter(6, 0.1, btype='lowpass')
    filtered_signal = signal.filtfilt(b, a, audio_data)
    wavfile.write(os.path.join(output_directory, 'butter.wav'), sample_rate, filtered_signal.astype(np.int16))
    draw_spectrogram(filtered_signal, sample_rate, os.path.join(output_directory, 'butter.png'))

def apply_savgol_filter(sample_rate, audio_data, output_directory):
    denoised_savgol = signal.savgol_filter(audio_data, 75, 5)
    wavfile.write(os.path.join(output_directory, 'savgol.wav'), sample_rate, denoised_savgol.astype(np.int16))
    draw_spectrogram(denoised_savgol, sample_rate, os.path.join(output_directory, 'savgol.png'))

def find_peaks(sample_rate, audio_data, output_directory):
    peaks = set()
    delta_time = 0.1
    delta_frequency = 50

    frequencies, time, spectrogram = draw_spectrogram(audio_data, sample_rate, os.path.join(output_directory, 'input.png'))

    for i in range(len(frequencies)):
        for j in range(len(time)):
            index_time = np.asarray(abs(time - time[j]) < delta_time).nonzero()[0]
            index_frequency = np.asarray(abs(frequencies - frequencies[i]) < delta_frequency).nonzero()[0]
            indexes = np.array([x for x in itertools.product(index_frequency, index_time)])
            flag = True
            for a, b in indexes:
                if spectrogram[i, j] <= spectrogram[a, b] and i != a and i != b:
                    flag = False
                    break

            if flag:
                peaks.add(time[j])

    with open(os.path.join(output_directory, 'peaks.txt'), 'w') as f:
        f.write(str(len(peaks)))
        f.write('\n')
        f.write(str(peaks))

def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_directory, 'input')
    output_path = os.path.join(current_directory, 'output')
    os.makedirs(output_path, exist_ok=True)

    sample_rate, audio_data = wavfile.read(os.path.join(input_path, 'audio_file.wav'))

    apply_butterworth_filter(sample_rate, audio_data, output_path)
    print('Butterworth filter applied')

    apply_savgol_filter(sample_rate, audio_data, output_path)
    print('Savitzky-Golay filter applied')

    find_peaks(sample_rate, audio_data, output_path)

if __name__ == '__main__':
    main()