import ffmpeg

def convert_audio_ffmpeg(input_file, output_file):
    try:
        ffmpeg.input(input_file).output(output_file).run()
        print(f"File berhasil dikonversi ke {output_file}")
    except ffmpeg.Error as e:
        print("Terjadi kesalahan saat konversi audio:", e)

# Contoh penggunaan
convert_audio_ffmpeg("audio-noise.aac", "audio-nonoise.wav")
