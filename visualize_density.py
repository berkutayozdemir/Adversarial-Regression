import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset import load_data
from src.model import Generator

def visualize_density():
    print("1. Loading Data...")
    BATCH_SIZE = 2000
    _, test_loader, input_dim = load_data(batch_size=BATCH_SIZE)
    Z_DIM = 10
    
    print("2. Loading Model...")
    generator = Generator(input_dim, z_dim=Z_DIM)
    try:
        generator.load_state_dict(torch.load('gan_generator.pth'))
        print("   Model loaded successfully.")
    except FileNotFoundError:
        print("   Error: 'gan_generator.pth' not found. Please run run_pipeline.py first.")
        return

    generator.eval()
    
    # Generate Predictions
    print("3. Generating Predictions...")
    all_real_y = []
    all_fake_y = []
    
    with torch.no_grad():
        for real_x, real_y in test_loader:
            z = torch.randn(real_x.size(0), Z_DIM)
            fake_y = generator(real_x, z)
            
            all_real_y.append(real_y.numpy())
            all_fake_y.append(fake_y.numpy())
            
    y_true_np = np.concatenate(all_real_y).flatten()
    y_fake_np = np.concatenate(all_fake_y).flatten()
    
    # --- Plotting ---
    print("4. Creating Density Plots...")
    plt.figure(figsize=(10, 6))
    
    # KDE Plot
    sns.kdeplot(y_true_np, fill=True, label='Gercek Veri Yoğunluğu', color='blue', alpha=0.3)
    sns.kdeplot(y_fake_np, fill=True, label='Uretilen Veri Yoğunluğu', color='purple', alpha=0.3)
    
    plt.title('Gercek Veri ve Uretilen Veri Yoğunlugu')
    plt.xlabel('Hedef Değer')
    plt.ylabel('Yoğunluk')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('density_results.png')
    print("   Saved plot to density_results.png")
    
    # --- Optional: Histogram ---
    plt.figure(figsize=(10, 6))
    plt.hist(y_true_np, bins=50, alpha=0.5, label='Gercek Veri', density=True, color='blue')
    plt.hist(y_fake_np, bins=50, alpha=0.5, label='Uretilen Veri', density=True, color='purple')
    plt.title('Gercek Veri ve Uretilen Veri Histogrami')
    plt.legend()
    plt.savefig('histogram_results.png')
    print("   Saved plot to histogram_results.png")

if __name__ == "__main__":
    visualize_density()
