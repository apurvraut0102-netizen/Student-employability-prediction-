# 🎓 Student Employability Prediction — Neha Lutade

A college-level Machine Learning project that predicts student employability from academic performance, attendance, internships, projects, certifications, communication, technical skills, aptitude, backlogs and work experience.

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Random Forest Classifier

## Model Performance
The included model was evaluated on a held-out test set. The current synthetic dataset gives approximately **93.3% test accuracy**. Exact results can change if the dataset is replaced or regenerated.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```text
neha_lutade_employability/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── student_employability.csv
└── model/
    └── employability_model.pkl
```

## Important
The included dataset is **synthetic/educational data**, created for a college demonstration. The prediction should not be used for real hiring decisions.

## Author
**Neha Lutade**
