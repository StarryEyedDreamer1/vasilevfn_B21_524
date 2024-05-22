import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import maximum_filter

def determine_max_frequency(filepath):
    data, sample_rate = librosa.load(filepath)
    chroma = librosa.feature.chroma_stft(y=data, sr=sample_rate)
    f0 = librosa.piptrack(y=data, sr=sample_rate, S=chroma)[0]
    max_f0_index = np.argmax(f0)
    return max_f0_index

def plot_spectrogram(samples, sample_rate, filepath):
    frequencies, time, spec = signal.spectrogram(samples, sample_rate, window=('hann'))
    spec = np.log10(spec + 1)
    plt.pcolormesh(time, frequencies, spec, shading='gouraud', vmin=spec.min(), vmax=spec.max())
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.savefig(filepath)
    return frequencies, time, spec

def find_top_peaks(frequencies, time, spec):
    delta_time = int(0.1 * len(time))
    delta_freq = int(50 / (frequencies[1] - frequencies[0]))
    filtered = maximum_filter(spec, size=(delta_freq, delta_time))

    peaks_mask = (spec == filtered)
    peak_values = spec[peaks_mask]
    peak_frequencies = frequencies[peaks_mask.any(axis=1)]

    top_indices = np.argsort(peak_values)[-3:]
    top_frequencies = peak_frequencies[top_indices]

    return list(top_frequencies)

def determine_max_min_frequency(voice_path):
    y, sr = librosa.load(voice_path, sr=None)

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    frequencies = librosa.fft_frequencies(sr=sr)
    mean_spec = np.mean(D, axis=1)

    idx_min = np.argmax(mean_spec > -80)
    idx_max = len(mean_spec) - np.argmax(mean_spec[::-1] > -80) - 1

    min_freq = frequencies[idx_min]
    max_freq = frequencies[idx_max]

    return max_freq, min_freq

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'output')
    os.makedirs(output_path, exist_ok=True)

    input_path = os.path.join(current_dir, 'input')
    humiliations = ['word_a', 'word_i', 'sound']
    humiliation_voice_paths = [
        (humiliation, os.path.join(input_path, f'{humiliation}.wav'))
        for humiliation in humiliations
    ]
    with open(os.path.join(output_path, 'result.txt'), 'w') as result_file:
        for humiliation, voice_path in humiliation_voice_paths:
            rate, samples = wavfile.read(voice_path)
            frequencies, time, spec = plot_spectrogram(samples, rate, os.path.join(output_path, f'{humiliation}.png'))
            max_freq, min_freq = determine_max_min_frequency(voice_path)

            result_file.write(f'{humiliation}:\n')
            result_file.write(f'\tMaximum Frequency: {max_freq}\n')
            result_file.write(f'\tMinimum Frequency: {min_freq}\n')
            result_file.write(f"\tMost Timbrally Colored Fundamental Frequency: {determine_max_frequency(voice_path)}. "
                              "This is the frequency with the most overtones\n")
            if 'word' in humiliation:
                result_file.write(f"\tThree Strongest Formants: {find_top_peaks(frequencies, time, spec)}. "
                                  "These are frequencies with the highest energy in some neighborhood\n")

if __name__ == "__main__":
    main()

