import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import optuna
import joblib

def clean_numeric(x):
    """
    Tisztítja és konvertálja a numerikus értékeket string formátumból.
    
    Args:
        x: A konvertálandó érték
        
    Returns:
        float: A konvertált numerikus érték vagy np.nan ha nem sikerült a konverzió
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        try:
            # Eltávolítjuk a mértékegységeket (bhp, cc, kmpl, stb.)
            x = x.lower().replace('bhp', '').replace('cc', '').replace('kmpl', '').strip()
            # Csak a számokat és a pontot tartjuk meg
            numeric_str = ''.join([c for c in x if c.isdigit() or c == '.'])
            return float(numeric_str) if numeric_str else np.nan
        except:
            return np.nan
    return float(x) if not pd.isna(x) else np.nan

# Adatok betöltése és előfeldolgozása
def load_and_preprocess_data():
    # Adatok betöltése
    train_data = pd.read_csv('train-data.csv')
    test_data = pd.read_csv('test-data.csv')
    
    print("\nAdathalmaz információk:")
    print("\nTanító adathalmaz alakja:", train_data.shape)
    print("\nTeszt adathalmaz alakja:", test_data.shape)
    
    # Power, Engine és Mileage oszlopok tisztítása
    numeric_columns = ['Power', 'Engine', 'Mileage']
    for column in numeric_columns:
        train_data[column] = train_data[column].apply(clean_numeric)
        test_data[column] = test_data[column].apply(clean_numeric)
        
        # Kiírjuk az oszlopok statisztikáit a hibakereséshez
        print(f"\n{column} oszlop statisztikák:")
        print(f"Hiányzó értékek száma (train): {train_data[column].isna().sum()}")
        print(f"Egyedi értékek száma (train): {train_data[column].nunique()}")
        print(f"Minimum érték (train): {train_data[column].min()}")
        print(f"Maximum érték (train): {train_data[column].max()}")
    
    # Feature engineering
    # Kor számítása
    current_year = 2024
    train_data['Age'] = current_year - train_data['Year']
    test_data['Age'] = current_year - test_data['Year']
    
    # Kilométeróra állás per év - végtelen értékek kezelése
    train_data['Km_Per_Year'] = train_data['Kilometers_Driven'] / train_data['Age'].replace(0, 1)
    test_data['Km_Per_Year'] = test_data['Kilometers_Driven'] / test_data['Age'].replace(0, 1)
    
    # Power/Engine arány - nullával való osztás kezelése
    train_data['Power_per_Engine'] = train_data['Power'] / train_data['Engine'].replace(0, np.nan)
    test_data['Power_per_Engine'] = test_data['Power'] / test_data['Engine'].replace(0, np.nan)
    
    # Kategorikus változók kódolása
    categorical_columns = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type']
    label_encoders = {}
    
    for column in categorical_columns:
        # Először kezeljük a hiányzó értékeket a leggyakoribb értékkel
        most_frequent = train_data[column].mode()[0]
        train_data[column] = train_data[column].fillna(most_frequent)
        test_data[column] = test_data[column].fillna(most_frequent)
        
        label_encoders[column] = LabelEncoder()
        train_data[column] = label_encoders[column].fit_transform(train_data[column])
        
        # Az ismeretlen kategóriákat a leggyakoribb kategóriára térképezzük
        known_categories = set(label_encoders[column].classes_)
        test_data[column] = test_data[column].map(lambda x: x if x in known_categories else most_frequent)
        test_data[column] = label_encoders[column].transform(test_data[column])
    
    # Brand kezelése
    train_data['Brand'] = train_data['Name'].fillna('Other').apply(lambda x: x.split()[0] if isinstance(x, str) else 'Other')
    test_data['Brand'] = test_data['Name'].fillna('Other').apply(lambda x: x.split()[0] if isinstance(x, str) else 'Other')
    
    # Csak azokat a márkákat tartjuk meg, amelyek legalább 5-ször előfordulnak
    brand_counts = train_data['Brand'].value_counts()
    frequent_brands = brand_counts[brand_counts >= 5].index
    train_data['Brand'] = train_data['Brand'].map(lambda x: x if x in frequent_brands else 'Other')
    test_data['Brand'] = test_data['Brand'].map(lambda x: x if x in frequent_brands else 'Other')
    
    brand_encoder = LabelEncoder()
    train_data['Brand'] = brand_encoder.fit_transform(train_data['Brand'])
    test_data['Brand'] = brand_encoder.transform(test_data['Brand'])
    
    # Brand átlagos ára
    brand_mean_price = train_data.groupby('Brand')['Price'].mean()
    train_data['Brand_Mean_Price'] = train_data['Brand'].map(brand_mean_price)
    test_data['Brand_Mean_Price'] = test_data['Brand'].map(brand_mean_price)
    
    # Hiányzó értékek kezelése
    numerical_columns = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats',
                        'Age', 'Km_Per_Year', 'Power_per_Engine', 'Brand_Mean_Price']
    
    # Hiányzó értékek kezelése medián értékekkel
    for column in numerical_columns:
        median_value = train_data[column].median()
        train_data[column] = train_data[column].fillna(median_value)
        test_data[column] = test_data[column].fillna(median_value)
    
    # Felesleges oszlopok eltávolítása
    columns_to_drop = ['Unnamed: 0', 'New_Price', 'Name', 'Year']  # Year helyett Age-et használunk
    train_data.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')
    test_data.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')
    
    # Adatok vizualizációja
    plot_data_analysis(train_data, label_encoders, brand_encoder)
    
    # Jellemzők és célváltozó szétválasztása
    X = train_data.drop('Price', axis=1)
    y = train_data['Price']
    
    return X, y, test_data, numerical_columns, categorical_columns

def plot_data_analysis(data, label_encoders, brand_encoder):
    # Korrelációs mátrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Korrelációs mátrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Ár eloszlása
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Price', bins=50)
    plt.title('Árak eloszlása')
    plt.xlabel('Ár (lakhs)')
    plt.ylabel('Gyakoriság')
    plt.tight_layout()
    plt.savefig('price_distribution.png')
    plt.close()
    
    # Brand szerinti átlagár
    plt.figure(figsize=(15, 6))
    # Visszakódoljuk a márkaneveket
    brand_names = brand_encoder.inverse_transform(range(len(brand_encoder.classes_)))
    brand_avg_price = data.groupby('Brand')['Price'].mean()
    brand_avg_price.index = brand_names[brand_avg_price.index]
    brand_avg_price = brand_avg_price.sort_values(ascending=False).head(15)
    
    sns.barplot(x=brand_avg_price.index, y=brand_avg_price.values)
    plt.title('Top 15 márka átlagára')
    plt.xticks(rotation=45)
    plt.xlabel('Márka')
    plt.ylabel('Átlagár (lakhs)')
    plt.tight_layout()
    plt.savefig('brand_avg_price.png')
    plt.close()
    
    # Kor vs Ár
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Age', y='Price', alpha=0.5)
    plt.title('Kor és ár kapcsolata')
    plt.xlabel('Kor (év)')
    plt.ylabel('Ár (lakhs)')
    plt.tight_layout()
    plt.savefig('age_vs_price.png')
    plt.close()
    
    # Kilométeróra állás vs Ár
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Kilometers_Driven', y='Price', alpha=0.5)
    plt.title('Kilométeróra állás és ár kapcsolata')
    plt.xlabel('Kilométeróra állás')
    plt.ylabel('Ár (lakhs)')
    plt.tight_layout()
    plt.savefig('km_vs_price.png')
    plt.close()
    
    # Power vs Ár
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Power', y='Price', alpha=0.5)
    plt.title('Teljesítmény és ár kapcsolata')
    plt.xlabel('Teljesítmény (bhp)')
    plt.ylabel('Ár (lakhs)')
    plt.tight_layout()
    plt.savefig('power_vs_price.png')
    plt.close()
    
    # Üzemanyag típus szerinti átlagár
    plt.figure(figsize=(10, 6))
    # Visszakódoljuk az üzemanyag típusokat
    fuel_names = label_encoders['Fuel_Type'].inverse_transform(range(len(label_encoders['Fuel_Type'].classes_)))
    fuel_avg_price = data.groupby('Fuel_Type')['Price'].mean()
    fuel_avg_price.index = fuel_names[fuel_avg_price.index]
    fuel_avg_price = fuel_avg_price.sort_values(ascending=False)
    
    sns.barplot(x=fuel_avg_price.index, y=fuel_avg_price.values)
    plt.title('Üzemanyag típus szerinti átlagár')
    plt.xticks(rotation=45)
    plt.xlabel('Üzemanyag típus')
    plt.ylabel('Átlagár (lakhs)')
    plt.tight_layout()
    plt.savefig('fuel_type_avg_price.png')
    plt.close()
    
    # Tulajdonosok száma szerinti átlagár
    plt.figure(figsize=(10, 6))
    # Visszakódoljuk a tulajdonos típusokat
    owner_names = label_encoders['Owner_Type'].inverse_transform(range(len(label_encoders['Owner_Type'].classes_)))
    owner_avg_price = data.groupby('Owner_Type')['Price'].mean()
    owner_avg_price.index = owner_names[owner_avg_price.index]
    owner_avg_price = owner_avg_price.sort_values(ascending=False)
    
    sns.barplot(x=owner_avg_price.index, y=owner_avg_price.values)
    plt.title('Tulajdonosok száma szerinti átlagár')
    plt.xlabel('Tulajdonosok száma')
    plt.ylabel('Átlagár (lakhs)')
    plt.tight_layout()
    plt.savefig('owner_type_avg_price.png')
    plt.close()

class CarPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(CarPricePredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def objective(trial, X, y):
    # Hyperparaméterek definiálása
    hidden_sizes = [
        trial.suggest_int(f'hidden_size_1', 32, 256),
        trial.suggest_int(f'hidden_size_2', 16, 128),
        trial.suggest_int(f'hidden_size_3', 8, 64)
    ]
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Modell létrehozása és tanítása
        model = CarPricePredictor(X.shape[1], hidden_sizes, dropout_rate)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor.reshape(-1, 1))
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor.reshape(-1, 1))
            cv_scores.append(val_loss.item())
    
    return np.mean(cv_scores)

def train_ensemble(X, y, test_data, best_params):
    # Adatok normalizálása
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
    X_test_scaled = scaler_X.transform(test_data)
    
    # Ensemble modellek létrehozása
    models = []
    n_models = 5
    
    for i in range(n_models):
        # Neural Network
        nn_model = CarPricePredictor(
            X.shape[1],
            hidden_sizes=[
                best_params[f'hidden_size_1'],
                best_params[f'hidden_size_2'],
                best_params[f'hidden_size_3']
            ],
            dropout_rate=best_params['dropout_rate']
        )
        
        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42 + i
        )
        
        models.extend([nn_model, rf_model])
    
    # Modellek tanítása
    predictions = []
    
    for i, model in enumerate(models):
        if isinstance(model, nn.Module):
            # Neural Network tanítása
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=best_params['learning_rate'],
                weight_decay=best_params['weight_decay']
            )
            
            # Learning rate scheduler hozzáadása - verbose paraméter nélkül
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                min_lr=1e-6
            )
            
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y_scaled)
            
            # Train-validation split
            train_size = int(0.8 * len(X_tensor))
            indices = torch.randperm(len(X_tensor))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            X_train = X_tensor[train_indices]
            y_train = y_tensor[train_indices]
            X_val = X_tensor[val_indices]
            y_val = y_tensor[val_indices]
            
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 10
            
            for epoch in range(100):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train)
                train_loss = criterion(outputs, y_train.reshape(-1, 1))
                train_loss.backward()
                optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    val_loss = criterion(val_outputs, y_val.reshape(-1, 1))
                    
                    # Learning rate scheduler update
                    scheduler.step(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= max_patience:
                        print(f"Early stopping triggered for model {i} at epoch {epoch}")
                        break
                    
                    # Kiírjuk a tanulási folyamat állapotát minden 10. epochban
                    if epoch % 10 == 0:
                        print(f"Model {i}, Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                              f"LR = {optimizer.param_groups[0]['lr']:.6f}")
            
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test_scaled)
                pred = model(X_test_tensor).numpy()
        else:
            # Random Forest tanítása
            model.fit(X_scaled, y_scaled)
            pred = model.predict(X_test_scaled).reshape(-1, 1)
        
        predictions.append(pred)
    
    # Ensemble predikciók átlagolása
    ensemble_predictions = np.mean(predictions, axis=0)
    final_predictions = scaler_y.inverse_transform(ensemble_predictions)
    
    return final_predictions, models

def main():
    # Adatok betöltése és előfeldolgozása
    X, y, test_data, numerical_columns, categorical_columns = load_and_preprocess_data()
    
    # Hyperparaméter optimalizáció
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X.values, y.values), n_trials=50)
    
    best_params = study.best_params
    print("\nLegjobb hyperparaméterek:", best_params)
    
    # Ensemble modell tanítása
    predictions, models = train_ensemble(X, y, test_data, best_params)
    
    # Eredmények mentése
    test_predictions_df = pd.DataFrame(predictions, columns=['Predicted_Price'])
    test_predictions_df.to_csv('predictions.csv', index=False)
    print("\nA predikciók mentésre kerültek a 'predictions.csv' fájlba.")
    
    # Modellek mentése
    for i, model in enumerate(models):
        if isinstance(model, nn.Module):
            torch.save(model.state_dict(), f'nn_model_{i}.pth')
        else:
            joblib.dump(model, f'rf_model_{i}.joblib')
    
    # Feature importance vizualizáció Random Forest alapján
    rf_model = models[1]  # Első Random Forest modell
    feature_importance = rf_model.feature_importances_
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xticks(range(len(feature_importance)), X.columns, rotation=45, ha='right')
    plt.xlabel('Jellemzők')
    plt.ylabel('Fontosság')
    plt.title('Jellemzők fontossága (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance_rf.png')
    plt.close()

if __name__ == "__main__":
    main() 