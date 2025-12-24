import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("student_performance")

def build_model():
    mlflow.autolog()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    X = pd.read_csv(os.path.join(BASE_DIR, "Student_Performance_Preprocessing", "X.csv"))
    y = pd.read_csv(os.path.join(BASE_DIR, "Student_Performance_Preprocessing", "y.csv"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="student_performance"):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(registered_model_name="student_performance_model", sk_model=model, artifact_path="model")

if __name__ == "__main__":
    build_model()