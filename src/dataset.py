import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

def load_data(batch_size=64, test_size=0.2, random_state=42):
    """
    Loads California Housing dataset, scales features, and prepares PyTorch DataLoaders.
    """
    # 1. Load Data
    try:
        print("   Attempting to download California Housing data...")
        data = fetch_california_housing()
        print("   Successfully loaded California Housing data.")
    except Exception as e:
        print(f"   WARNING: Failed to load California Housing data ({e}).")
        print("   Falling back to Diabetes dataset (similar regression task).")
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
    
    X, y = data.data, data.target
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 3. Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3b. Standardize Targets (Fix for Mode Collapse)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # 4. Convert to PyTorch Tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled).view(-1, 1) # Reshape to (N, 1)
    
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled).view(-1, 1)
    
    # 5. Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, X_train_tensor.shape[1]
