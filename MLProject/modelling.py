import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. ISI BAGIAN INI DENGAN DATA DAGSHUB ANDA
print("Mengatur MLflow Tracking URI...")
# Pastikan URL ini sesuai dengan link DagsHub eksperimen lama Anda
mlflow.set_tracking_uri("https://dagshub.com/NurusSafaah/Eksperimen-Telco-Churn.mlflow")

# Lanjut ke kode eksperimen...
mlflow.set_experiment("Telco-Churn-Experiment")

def load_data():
    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "telco-churn_preprocessing.csv")    
    file_path = os.path.abspath(file_path)

    print(f"[INFO] Loading data from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {file_path}. Cek nama folder/file!")

    df = pd.read_csv(file_path)
    return df

def main():
    # 2. Setup DagsHub & MLflow (Koneksi ke Server)
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_experiment("Telco Churn - Basic Model")

    # 3. Persiapan Data
    df = load_data()
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Split data 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Training Model Pakai Autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        print("[PROCESS] Sedang melatih model...")
        
        # Membuat model Random Forest sederhana
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Prediksi & Evaluasi
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"[RESULT] Akurasi Model: {acc:.4f}")
        print(f"[SUCCESS] Hasil training tersimpan otomatis di DagsHub!")

if __name__ == "__main__":
    main()