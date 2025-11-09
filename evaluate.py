import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import load_preprocessed_data, GENRES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model():
    """Evaluate PyTorch model"""
    
    # Load data
    print("📂 Loading test data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
    
    print(f"Original shape - X_test: {X_test.shape}")
    
    # Normalize test data to [-1, 1] range (SAME AS TRAINING)
    X_train_min = X_train.min()
    X_train_max = X_train.max()
    
    X_test = 2 * (X_test - X_train_min) / (X_train_max - X_train_min) - 1
    
    # Reshape for CNN - add channel dimension
    # From (200, 128, 1292) to (200, 1, 128, 1292)
    X_test = X_test[:, np.newaxis, :, :]
    
    print(f"Reshaped X_test: {X_test.shape}")
    
    X_test_torch = torch.FloatTensor(X_test).to(device)
    
    # Load model
    print("Using device: cuda" if torch.cuda.is_available() else "Using device: cpu")
    from train import MusicGenreNet
    model = MusicGenreNet(num_classes=len(GENRES)).to(device)
    model.load_state_dict(torch.load('models/genre_model.pth', map_location=device))
    model.eval()
    
    print("✅ Model loaded")
    
    # Predictions
    print("🎯 Making predictions...")
    with torch.no_grad():
        outputs = model(X_test_torch)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
    
    # Accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\n✅ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=GENRES))
    
    # Per-genre accuracy
    print("\n📊 Per-Genre Accuracy:")
    print("-" * 60)
    cm = confusion_matrix(y_test, y_pred)
    for idx, genre in enumerate(GENRES):
        genre_acc = cm[idx, idx] / cm[idx].sum()
        print(f"  {genre:12s}: {genre_acc:.4f} ({genre_acc*100:.2f}%)")
    print("-" * 60)
    
    # Confusion matrix visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=GENRES, 
        yticklabels=GENRES,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Music Genre Classification', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✅ Confusion matrix saved to results/confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    evaluate_model()
