import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_prep import load_and_clean_data

def train_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'train.csv')
    model_path = os.path.join(base_dir, 'model.pkl')
    scaler_path = os.path.join(base_dir, 'scaler.pkl')

    print("Завантаження даних...")
    df = load_and_clean_data(data_path)

    X = df.drop('price_range', axis=1)
    y = df['price_range']

    print("Розбиття та масштабування...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Навчання моделі...")
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)

    y_pred = log_reg.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Точність на тестовій вибірці: {acc * 100:.2f}%")

    joblib.dump(log_reg, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Артефакти успішно збережено в корінь проєкту!")

if __name__ == "__main__":
    train_pipeline()