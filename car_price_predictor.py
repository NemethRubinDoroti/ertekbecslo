import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Adatok betöltése és előfeldolgozása
def load_and_preprocess_data():
    # Adatok betöltése
    train_data = pd.read_csv('train-data.csv')
    test_data = pd.read_csv('test-data.csv')
    
    print("\nAdathalmaz információk:")
    print("\nTanító adathalmaz alakja:", train_data.shape)
    print("\nTeszt adathalmaz alakja:", test_data.shape)
    print("\nOszlopok:")
    print(train_data.columns.tolist())
    print("\nElső néhány sor:")
    print(train_data.head())
    print("\nStatisztikai összefoglaló:")
    print(train_data.describe())
    print("\nHiányzó értékek:")
    print(train_data.isnull().sum())
    
    # Numerikus oszlopok tisztítása és konvertálása
    def clean_numeric(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            # Csak az első számot vesszük figyelembe
            try:
                return float(''.join([c for c in x.split()[0] if c.isdigit() or c == '.']))
            except:
                return np.nan
        return float(x)
    
    # Power, Engine és Mileage oszlopok tisztítása
    for column in ['Power', 'Engine', 'Mileage']:
        train_data[column] = train_data[column].apply(clean_numeric)
        test_data[column] = test_data[column].apply(clean_numeric)
    
    # Kategorikus változók kódolása
    categorical_columns = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type']
    label_encoders = {}
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        train_data[column] = label_encoders[column].fit_transform(train_data[column])
        # Kezeljük az ismeretlen kategóriákat a teszt adathalmazban
        known_categories = set(label_encoders[column].classes_)
        test_data[column] = test_data[column].map(lambda x: x if x in known_categories else list(known_categories)[0])
        test_data[column] = label_encoders[column].transform(test_data[column])
    
    # Name oszlop kezelése - csak a márka használata
    train_data['Brand'] = train_data['Name'].apply(lambda x: x.split()[0])
    test_data['Brand'] = test_data['Name'].apply(lambda x: x.split()[0])
    
    brand_encoder = LabelEncoder()
    train_data['Brand'] = brand_encoder.fit_transform(train_data['Brand'])
    # Kezeljük az ismeretlen márkákat
    known_brands = set(brand_encoder.classes_)
    test_data['Brand'] = test_data['Brand'].map(lambda x: x if x in known_brands else list(known_brands)[0])
    test_data['Brand'] = brand_encoder.transform(test_data['Brand'])
    
    # Hiányzó értékek kezelése
    numerical_columns = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats']
    for column in numerical_columns:
        if train_data[column].isnull().sum() > 0:
            mean_value = train_data[column].mean()
            train_data[column] = train_data[column].fillna(mean_value)
        if test_data[column].isnull().sum() > 0:
            mean_value = train_data[column].mean()
            test_data[column] = test_data[column].fillna(mean_value)
    
    # New_Price és Name oszlopok eltávolítása
    train_data.drop(['Unnamed: 0', 'New_Price', 'Name'], axis=1, inplace=True)
    test_data.drop(['Unnamed: 0', 'New_Price', 'Name'], axis=1, inplace=True)
    
    # Jellemzők és célváltozó szétválasztása
    X = train_data.drop('Price', axis=1)
    y = train_data['Price']
    
    # Adatok normalizálása
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    # Tanító és validációs adatok szétválasztása
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # Teszt adatok előkészítése
    X_test_scaled = scaler_X.transform(test_data)
    
    return (X_train, X_val, y_train, y_val, X_test_scaled, scaler_y,
            X_train.shape[1], X.columns)

# Neurális hálózat modell
class CarPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(CarPricePredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer4(x)
        return x

# Modell tanítása
def train_model(X_train, X_val, y_train, y_val, input_size, num_epochs=200):
    # Adatok átalakítása PyTorch tenzorokká
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    # Modell létrehozása
    model = CarPricePredictor(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Tanítási és validációs veszteségek tárolása
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    # Tanítási ciklus
    for epoch in range(num_epochs):
        # Tanítási fázis
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.reshape(-1, 1))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validációs fázis
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val.reshape(-1, 1))
            val_losses.append(val_loss.item())
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    # Tanulási görbék megjelenítése
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('learning_curves.png')
    plt.close()
    
    return model

def main():
    # Adatok betöltése és előfeldolgozása
    X_train, X_val, y_train, y_val, X_test_scaled, scaler_y, input_size, feature_names = load_and_preprocess_data()
    
    # Modell tanítása
    model = train_model(X_train, X_val, y_train, y_val, input_size)
    
    # Predikciók készítése a teszt adatokra
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        test_predictions_scaled = model(X_test_tensor)
        test_predictions = scaler_y.inverse_transform(test_predictions_scaled)
    
    # Eredmények mentése
    test_predictions_df = pd.DataFrame(test_predictions, columns=['Predicted_Price'])
    test_predictions_df.to_csv('predictions.csv', index=False)
    print("\nA predikciók mentésre kerültek a 'predictions.csv' fájlba.")
    
    # Feature importance vizualizáció (egyszerű módszer a súlyok alapján)
    feature_importance = np.abs(model.layer1.weight.data.numpy()).mean(axis=0)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xticks(range(len(feature_importance)), feature_names, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance Based on Neural Network Weights')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

if __name__ == "__main__":
    main() 