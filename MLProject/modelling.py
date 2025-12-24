import setuptools
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import argparse
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_model(data_path, n_estimators):
    print(f"Memuat dataset dari: {data_path}")
    df = pd.read_csv(data_path)
    
    target = 'mental_state'
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.sklearn.autolog()
    
    print(f"Memulai training (n_estimators={n_estimators})...")

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr', average='weighted')
        except:
            roc_auc = 0.0

        print("Mencatat metrik test set...")
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)

        mlflow.sklearn.log_model(model, "model")
        
        os.makedirs("output", exist_ok=True)
        joblib.dump(model, "output/model.joblib")
        print("Model selesai dilatih.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="mh_sosmed_dataset.csv", help="Path ke dataset")
    parser.add_argument("--n_estimators", type=int, default=100, help="Jumlah trees di Random Forest")
    args = parser.parse_args()

    train_model(args.data_file, args.n_estimators)