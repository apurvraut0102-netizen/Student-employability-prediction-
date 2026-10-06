
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT / "data" / "student_employability.csv")
X = df.drop(columns=["Employability"])
y = df["Employability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=250, max_depth=9, min_samples_leaf=2,
        random_state=42, class_weight="balanced"
    ))
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred))

with open(ROOT / "model" / "employability_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Saved model to model/employability_model.pkl")
