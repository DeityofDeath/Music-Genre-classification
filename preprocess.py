import os
import librosa
import numpy as np
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'sample_rate': 22050,
    'n_fft': 2048,
    'hop_length': 512,
    'n_mels': 128,
    'n_mfcc': 13,
    'duration': 30  # seconds
}

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

class AudioPreprocessor:
    def __init__(self, config):
        self.config = config
        self.sr = config['sample_rate']
        
    def extract_mel_spectrogram(self, audio_path):
        """Extract mel-spectrogram from audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.config['duration'])
            
            # Pad if too short
            if len(y) < self.sr * self.config['duration']:
                y = np.pad(y, (0, self.sr * self.config['duration'] - len(y)))
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=self.config['n_fft'],
                hop_length=self.config['hop_length'],
                n_mels=self.config['n_mels']
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return mel_spec_db
        
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None
    
    def extract_mfcc(self, audio_path):
        """Extract MFCC features (alternative)"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.config['duration'])
            if len(y) < self.sr * self.config['duration']:
                y = np.pad(y, (0, self.sr * self.config['duration'] - len(y)))
            
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.config['n_mfcc'],
                n_fft=self.config['n_fft'],
                hop_length=self.config['hop_length']
            )
            
            return mfcc
        
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None


def prepare_dataset(data_dir, output_dir='spectrograms', feature_type='mel'):
    """Load GTZAN dataset and extract features"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    preprocessor = AudioPreprocessor(CONFIG)
    X = []
    y = []
    
    genre_to_idx = {genre: idx for idx, genre in enumerate(GENRES)}
    
    print("🎵 Extracting features from GTZAN dataset...")
    
    for genre in GENRES:
        genre_dir = os.path.join(data_dir, genre)
        
        if not os.path.exists(genre_dir):
            print(f"⚠️  Genre folder not found: {genre_dir}")
            continue
        
        audio_files = [f for f in os.listdir(genre_dir) if f.endswith('.wav')]
        
        for audio_file in tqdm(audio_files, desc=f"Processing {genre}"):
            audio_path = os.path.join(genre_dir, audio_file)
            
            if feature_type == 'mel':
                feature = preprocessor.extract_mel_spectrogram(audio_path)
            else:
                feature = preprocessor.extract_mfcc(audio_path)
            
            if feature is not None:
                X.append(feature)
                y.append(genre_to_idx[genre])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✅ Dataset shape: {X.shape}")
    print(f"✅ Labels shape: {y.shape}")
    
    # Save processed data
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    
    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump({'genres': GENRES, 'config': CONFIG}, f)
    
    return X, y


def load_preprocessed_data(output_dir='spectrograms', test_size=0.2, val_size=0.15):
    """Load preprocessed data and split into train/val/test"""
    
    X = np.load(os.path.join(output_dir, 'X.npy'))
    y = np.load(os.path.join(output_dir, 'y.npy'))
    
    # Normalize
    X = (X - X.mean()) / (X.std() + 1e-8)
    
    # Split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_size/(1-test_size), 
        random_state=42, 
        stratify=y_temp
    )
    
    print(f"✅ Train set: {X_train.shape}")
    print(f"✅ Val set: {X_val.shape}")
    print(f"✅ Test set: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Download GTZAN from Kaggle first:
    # kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
    
    data_dir = "data/genres_original"  # Path to GTZAN dataset
    
    # Preprocess
    X, y = prepare_dataset(data_dir, output_dir='spectrograms', feature_type='mel')
    
    # Load and split
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
    
    print("\n✨ Preprocessing complete!")
