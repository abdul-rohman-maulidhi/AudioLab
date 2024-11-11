import numpy as np
import librosa
import librosa.display
import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt

def noise_reduction(input_file, output_file, noise_reduction_level=0.2):
    # Load audio file
    y, sr = librosa.load(input_file, sr=None)
    
    # Convert to spectrogram
    stft = librosa.stft(y)
    spectrogram, phase = librosa.magphase(stft)
    
    # Average spectrum to estimate noise level
    noise_profile = np.mean(spectrogram, axis=1) * noise_reduction_level
    
    # Apply noise reduction by thresholding frequencies
    reduced_spectrogram = np.where(spectrogram > noise_profile[:, None], spectrogram, 0)
    
    # Reconstruct signal from spectrogram
    reduced_stft = reduced_spectrogram * phase
    y_denoised = librosa.istft(reduced_stft)
    
    # Save to output file
    wavfile.write(output_file, sr, (y_denoised * 32767).astype(np.int16))
    
    print(f"Noise-reduced audio saved to {output_file}")

# Example usage
noise_reduction('audio-bising.wav', 'output_audio.wav')
