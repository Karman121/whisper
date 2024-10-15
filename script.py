import whisper
import numpy as np

# Load the Whisper model
model = whisper.load_model("tiny")

# Load audio
audio = whisper.load_audio("audio_from_video.wav")

# Process in 30-second chunks
chunk_length = 30 * 16000  # 30 seconds * 16000 samples per second
total_chunks = len(audio) // chunk_length + (1 if len(audio) % chunk_length else 0)

for i in range(0, len(audio), chunk_length):
    chunk = audio[i:i+chunk_length]
    chunk = whisper.pad_or_trim(chunk)
    mel = whisper.log_mel_spectrogram(chunk).to(model.device)
    result = model.decode(mel, fp16=False)
    
    # Append the transcription to the file
    with open("transcription.txt", "a", encoding="utf-8") as f:
        f.write(result.text + "\n")
    
    # Print progress
    chunk_number = i // chunk_length + 1
    print(f"Processed chunk {chunk_number}/{total_chunks}")
    
    # Print the transcribed text for this chunk
    print(result.text)

print("Transcription complete!")
