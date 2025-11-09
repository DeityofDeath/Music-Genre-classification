import os
import numpy as np
import librosa
import torch
from preprocess import CONFIG, GENRES

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_genre(audio_path, model_path='models/genre_model.pth'):
    """Predict genre for a new audio file using PyTorch"""
    
    # Import model architecture
    from train import MusicGenreNet
    
    # Load model
    model = MusicGenreNet(num_classes=len(GENRES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Extract mel-spectrogram
    try:
        y, sr = librosa.load(audio_path, sr=CONFIG['sample_rate'], duration=CONFIG['duration'])
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return None, None
    
    # Pad if too short
    if len(y) < sr * CONFIG['duration']:
        y = np.pad(y, (0, sr * CONFIG['duration'] - len(y)))
    
    # Extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=CONFIG['n_fft'],
        hop_length=CONFIG['hop_length'],
        n_mels=CONFIG['n_mels']
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    # Add batch and channel dimensions
    mel_spec_db = torch.FloatTensor(mel_spec_db[np.newaxis, np.newaxis, ...]).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(mel_spec_db)
        probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
    
    genre_idx = np.argmax(probabilities)
    confidence = probabilities[genre_idx]
    
    # Display results
    print("\n" + "=" * 60)
    print(f"🎵 Audio file: {os.path.basename(audio_path)}")
    print(f"🎼 Predicted Genre: {GENRES[genre_idx]}")
    print(f"📊 Confidence: {confidence:.2%}")
    print("=" * 60)
    
    print(f"\n📋 All Predictions:")
    print("-" * 60)
    
    # Sort by confidence
    sorted_indices = np.argsort(probabilities)[::-1]
    for idx in sorted_indices:
        print(f"  {GENRES[idx]:12s}: {probabilities[idx]:.4f} ({probabilities[idx]*100:.2f}%)")
    
    print("=" * 60 + "\n")
    
    return GENRES[genre_idx], confidence


def batch_predict(audio_dir, model_path='models/genre_model.pth'):
    """Predict genres for all audio files in a directory"""
    
    results = []
    
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    
    print(f"🎵 Found {len(audio_files)} audio files in {audio_dir}\n")
    
    for audio_file in audio_files:
        audio_path = os.path.join(audio_dir, audio_file)
        genre, confidence = predict_genre(audio_path, model_path)
        
        if genre is not None:
            results.append({
                'file': audio_file,
                'predicted_genre': genre,
                'confidence': confidence
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 BATCH PREDICTION SUMMARY")
    print("=" * 60)
    
    for result in results:
        print(f"{result['file']:30s} → {result['predicted_genre']:12s} ({result['confidence']:.2%})")
    
    print("=" * 60 + "\n")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        
        if os.path.isfile(audio_path):
            # Single file prediction
            predict_genre(audio_path)
        elif os.path.isdir(audio_path):
            # Batch prediction
            batch_predict(audio_path)
        else:
            print(f"❌ Path not found: {audio_path}")
    else:
        print("Usage:")
        print("  Single file:  python predict.py path/to/song.wav")
        print("  Batch:        python predict.py path/to/audio/folder/")