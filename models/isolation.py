import librosa
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft
import soundfile as sf

def load_audio(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def separate_vocals_instruments(y, sr):
    # Compute the Short-Time Fourier Transform (STFT) of the audio signal
    f, t, Zxx = stft(y, fs=sr, nperseg=2048)
    
    # Separate magnitude and phase
    magnitude, phase = np.abs(Zxx), np.angle(Zxx)
    
    # Define frequency range typical for vocals (e.g., 80-1000 Hz)
    vocal_range = (f >= 80) & (f <= 1000)
    
    # Separate vocal and instrument components based on frequency range
    vocal_component = np.zeros_like(Zxx)
    instrumental_component = np.zeros_like(Zxx)
    
    # Mask frequencies within vocal range to isolate vocal component
    vocal_component[vocal_range, :] = Zxx[vocal_range, :]
    instrumental_component[~vocal_range, :] = Zxx[~vocal_range, :]
    
    # Reconstruct the vocal and instrumental tracks
    vocals = istft(vocal_component * np.exp(1j * phase), fs=sr)[1]
    instruments = istft(instrumental_component * np.exp(1j * phase), fs=sr)[1]
    
    return vocals, instruments

def save_audio(file_path, data, sr):
    # Save the separated audio files
    sf.write(file_path, data, sr)

def main():
    input_file = 'lagu.wav'  # Replace with your audio file
    vocals_output_file = 'vocals_output.wav'
    instruments_output_file = 'instruments_output.wav'
    
    # Load the audio
    y, sr = load_audio(input_file)
    
    # Separate vocals and instruments
    vocals, instruments = separate_vocals_instruments(y, sr)
    
    # Save the outputs
    save_audio(vocals_output_file, vocals, sr)
    save_audio(instruments_output_file, instruments, sr)
    
    print("Vocal and instrument separation complete. Files saved as:")
    print(f"Vocals: {vocals_output_file}")
    print(f"Instruments: {instruments_output_file}")

if __name__ == "__main__":
    main()
