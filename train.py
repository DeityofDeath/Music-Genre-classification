import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from preprocess import load_preprocessed_data, GENRES

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MusicGenreNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MusicGenreNet, self).__init__()
        
        # Conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        # Flatten size: 256 * 8 * 80 = 163840
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 8 * 80, 128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train_model():
    """Train PyTorch model"""
    
    # Load data
    print("📂 Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
    
    # Reshape for CNN - add channel dimension
    X_train = X_train[:, np.newaxis, :, :]
    X_val = X_val[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]
    
    print(f"Original data range - Min: {X_train.min():.4f}, Max: {X_train.max():.4f}, Mean: {X_train.mean():.4f}")
    
    # Normalize to [-1, 1] range (better for neural networks)
    X_train_min = X_train.min()
    X_train_max = X_train.max()
    X_train = 2 * (X_train - X_train_min) / (X_train_max - X_train_min) - 1
    X_val = 2 * (X_val - X_train_min) / (X_train_max - X_train_min) - 1
    X_test = 2 * (X_test - X_train_min) / (X_train_max - X_train_min) - 1
    
    print(f"Normalized data range - Min: {X_train.min():.4f}, Max: {X_train.max():.4f}, Mean: {X_train.mean():.4f}")
    print(f"✅ Train set: {X_train.shape}")
    print(f"✅ Val set: {X_val.shape}")
    print(f"✅ Test set: {X_test.shape}")
    
    # Convert to torch tensors
    X_train_torch = torch.FloatTensor(X_train).to(device)
    y_train_torch = torch.LongTensor(y_train).to(device)
    X_val_torch = torch.FloatTensor(X_val).to(device)
    y_val_torch = torch.LongTensor(y_val).to(device)
    X_test_torch = torch.FloatTensor(X_test).to(device)
    y_test_torch = torch.LongTensor(y_test).to(device)
    
    # Create data loaders with shuffling
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    val_dataset = TensorDataset(X_val_torch, y_val_torch)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Model
    print("\n🏗️  Building model...")
    model = MusicGenreNet(num_classes=len(GENRES)).to(device)
    print(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training
    print("\n🎯 Training model...")
    epochs = 100
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/genre_model.pth')
            print(f"  ✅ Best model saved (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()
    
    # Test
    print("\n📊 Testing on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=32):
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    print(f"✅ Test Accuracy: {test_acc:.2f}%")
    
    # Save results
    results = {
        'test_accuracy': float(test_acc),
        'epochs_trained': epoch + 1,
        'genres': GENRES,
        'best_val_loss': float(best_val_loss)
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot
    plot_training_history(history)
    
    return model, history


def plot_training_history(history):
    """Plot training history"""
    os.makedirs('results', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
    print("✅ Training history saved")
    plt.close()


if __name__ == "__main__":
    model, history = train_model()
    print("\n" + "=" * 60)
    print("✨ Training Complete!")
    print("=" * 60)
