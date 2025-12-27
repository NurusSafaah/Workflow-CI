import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# SETUP MLFLOW
# ==========================================
print("Mengatur MLflow Tracking URI...")

# URL DagsHub (Tetap Ada)
mlflow.set_tracking_uri("https://dagshub.com/NurusSafaah/Eksperimen-Telco-Churn.mlflow")

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "telco-churn_preprocessing.csv")    
    file_path = os.path.abspath(file_path)

    print(f"[INFO] Loading data from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {file_path}. Cek nama folder/file!")

    df = pd.read_csv(file_path)
    return df

def main():
    # 1. Persiapan Data
    df = load_data()
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Aktifkan Autolog
    # Autolog akan otomatis menempel pada "Run" yang dibuat oleh GitHub Actions
    mlflow.sklearn.autolog()
    
    print("[PROCESS] Sedang melatih model...")
    
    # 3. Training (Tanpa "with mlflow.start_run()")
    # Kita biarkan dia jalan di global context yang sudah dibuat oleh 'mlflow run'
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 4. Evaluasi
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"[RESULT] Akurasi Model: {acc:.4f}")
    print(f"[SUCCESS] Selesai! Log otomatis terkirim.")

if __name__ == "__main__":
    main()