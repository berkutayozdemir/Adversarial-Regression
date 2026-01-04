import torch
import numpy as np
import sklearn.metrics
from src.dataset import load_data
from src.model import Generator

def check_overfitting():
    print("1. Loading Data...")
    BATCH_SIZE = 64
    train_loader, test_loader, input_dim = load_data(batch_size=BATCH_SIZE)
    Z_DIM = 10
    
    print("2. Loading Model...")
    generator = Generator(input_dim, z_dim=Z_DIM)
    try:
        generator.load_state_dict(torch.load('gan_generator.pth'))
        print("   Model loaded successfully.")
    except FileNotFoundError:
        print("   Error: 'gan_generator.pth' not found.")
        return

    generator.eval()
    
    def evaluate(loader, name):
        all_real_y = []
        all_fake_y = []
        
        with torch.no_grad():
            for real_x, real_y in loader:
                # We generate ONE prediction per input to test regression capability
                # Ideally we might average over multiple noise samples, but single shot is standard for regression GAN inference
                z = torch.randn(real_x.size(0), Z_DIM)
                fake_y = generator(real_x, z)
                
                all_real_y.append(real_y.numpy())
                all_fake_y.append(fake_y.numpy())
                
        y_true = np.concatenate(all_real_y).flatten()
        y_pred = np.concatenate(all_fake_y).flatten()
        
        mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
        mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
        return mse, mae

    print("3. Calculating Metrics (Scaled Space)...")
    train_mse, train_mae = evaluate(train_loader, "Train")
    test_mse, test_mae  = evaluate(test_loader, "Test")
    
    print("-" * 40)
    print(f"Train MSE: {train_mse:.4f}  (MAE: {train_mae:.4f})")
    print(f"Test  MSE: {test_mse:.4f}  (MAE: {test_mae:.4f})")
    print("-" * 40)
    
    # Simple heuristic
    gap = test_mse - train_mse
    pct_gap = (gap / train_mse) * 100 if train_mse > 0 else 0
    
    print(f"Difference (Gap): {gap:.4f} ({pct_gap:.1f}%)")
    
    if test_mse > train_mse * 1.5: # Arbitrary threshold, >50% worse
        print("CONCLUSION: Potential Overfitting detected (Test MSE is significantly higher).")
    elif test_mse < train_mse:
        print("CONCLUSION: No Overfitting (Test error is lower or equal - likely underfitting or noise).")
    else:
        print("CONCLUSION: Balanced (Gap is within reasonable range).")
    print("-" * 40)

if __name__ == "__main__":
    check_overfitting()
