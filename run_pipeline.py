import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import load_data
from src.model import Generator, Discriminator

def main():
    # --- Hyperparameters ---
    BATCH_SIZE = 64
    EPOCHS = 5000
    LR = 0.0002        # 2e-4
    Z_DIM = 10
    D_UPDATES = 5  # Train Discriminator 5 times for every 1 Generator update
    
    print("1. Loading Data...")
    train_loader, test_loader, input_dim = load_data(batch_size=BATCH_SIZE)
    print(f"   Data Loaded. Input Dim: {input_dim}, Batch Size: {BATCH_SIZE}")
    
    # --- Initialize Models ---
    print("2. Initializing GAN Models...")
    generator = Generator(input_dim, z_dim=Z_DIM)
    discriminator = Discriminator(input_dim)
    
    # --- Optimizers ---
    # SGD with momentum 0.9 as requested
    optim_g = optim.SGD(generator.parameters(), lr=LR, momentum=0.9)
    optim_d = optim.SGD(discriminator.parameters(), lr=LR, momentum=0.9)
    
    # --- Loss ---
    criterion = nn.BCELoss() # Binary Cross Entropy
    
    print("3. Training GAN...")
    # Track losses
    g_losses = []
    d_losses = []
    
    for epoch in range(EPOCHS):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        for i, (real_x, real_y) in enumerate(train_loader):
            batch_size = real_x.size(0)
            
            # --- TRAIN DISCRIMINATOR (k times) ---
            # We train it more often to give it a chance to learn to reject fakes
            for _ in range(D_UPDATES):
                optim_d.zero_grad()
                
                # 1. Real Data
                # Discriminator should output 1 for real data
                pred_real = discriminator(real_x, real_y)
                target_real = torch.ones_like(pred_real)
                loss_d_real = criterion(pred_real, target_real)
                
                # 2. Fake Data
                # Generate fake y from noise
                z = torch.randn(batch_size, Z_DIM)
                fake_y = generator(real_x, z)
                
                # Discriminator should output 0 for fake data
                # Detach generator's output so we don't backprop into G here
                pred_fake = discriminator(real_x, fake_y.detach())
                target_fake = torch.zeros_like(pred_fake)
                loss_d_fake = criterion(pred_fake, target_fake)
                
                # Total D Loss
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optim_d.step()
            
            epoch_d_loss += loss_d.item()

            # --- TRAIN GENERATOR ---
            optim_g.zero_grad()
            
            # Generate fake y again (or reuse, but fresh noise is safer)
            z = torch.randn(batch_size, Z_DIM)
            fake_y = generator(real_x, z)
            
            # Generator wants Discriminator to output 1 (think it's real)
            pred_fake = discriminator(real_x, fake_y)
            target_real = torch.ones_like(pred_fake)
            
            loss_g = criterion(pred_fake, target_real)
            loss_g.backward()
            optim_g.step()
            
            epoch_g_loss += loss_g.item()
            
        # Log progress
        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        if (epoch + 1) % 500 == 0:
            print(f"   Epoch [{epoch+1}/{EPOCHS}]  Loss D: {avg_d_loss:.4f}  Loss G: {avg_g_loss:.4f}")
            
    print("5. Saving Model...")
    torch.save(generator.state_dict(), 'gan_generator.pth')
    print("Model saved to gan_generator.pth")

    print("6. Evaluating GAN Results...")
    # Visualize generated predictions vs real target
    generator.eval()
    
    all_real_y = []
    all_fake_y = []
    
    with torch.no_grad():
        for real_x, real_y in test_loader:
            z = torch.randn(real_x.size(0), Z_DIM)
            fake_y = generator(real_x, z)
            
            all_real_y.append(real_y.numpy())
            all_fake_y.append(fake_y.numpy())
            
    y_true_np = np.concatenate(all_real_y)
    y_fake_np = np.concatenate(all_fake_y)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true_np, y_fake_np, alpha=0.3, color='purple', label='Uretilen Veriler')
    # Ideal line
    min_val = min(y_true_np.min(), y_fake_np.min())
    max_val = max(y_true_np.max(), y_fake_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Identity (Perfect)')
    
    plt.xlabel('Gerçek Veriler')
    plt.ylabel('Uretilen Veriler')
    plt.title('Conditional GAN Regresyon Performansi')
    plt.legend()
    plt.grid(True)
    plt.savefig('gan_results.png')
    print("Saved plot to gan_results.png")

if __name__ == "__main__":
    main()
