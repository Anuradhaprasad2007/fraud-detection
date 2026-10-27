\# Fraud Detection in Online Payments



An end-to-end machine learning system that detects fraudulent transactions in real time, built on the PaySim synthetic mobile-money dataset. Includes a trained XGBoost model, a FastAPI backend with SHAP-based explainability, a PostgreSQL database for transaction history, and a Streamlit dashboard for live predictions.



\## Overview



Online payment fraud is rare (under 0.2% of transactions in this dataset) but costly, making it a classic imbalanced classification problem. This project trains a model to flag fraudulent transactions, explains \*why\* each prediction was made, and serves it through a working API and dashboard — not just a notebook.



\## Features



\- \*\*Fraud prediction API\*\* — FastAPI endpoint that classifies transactions as fraudulent or legitimate

\- \*\*Explainable predictions\*\* — SHAP values show which transaction features drove each decision

\- \*\*Persistent history\*\* — every prediction is logged to a PostgreSQL database

\- \*\*Interactive dashboard\*\* — Streamlit UI to test transactions and review recent predictions

\- \*\*Imbalanced-data handling\*\* — trained with `scale\_pos\_weight` and evaluated using precision/recall/PR-AUC rather than misleading accuracy



\## Tech Stack



| Layer | Technology |

|---|---|

| Model | XGBoost, scikit-learn |

| Backend | FastAPI, SQLAlchemy |

| Database | PostgreSQL (hosted on Neon) |

| Explainability | SHAP |

| Dashboard | Streamlit |

| Dataset | \[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) (synthetic mobile money transactions) |



\## Architecture



Streamlit Dashboard → FastAPI Backend → XGBoost Model (+ SHAP)

↓

PostgreSQL Database





\## Model Performance



Evaluated on a held-out test set (1.27M transactions, stratified split):



| Metric | Score |

|---|---|

| ROC-AUC | 0.999 |

| PR-AUC | 0.911 |

| Precision | 0.73 |

| Recall | 0.86 |

| F1 Score | 0.79 |



Decision threshold (0.90) was tuned on a separate validation split to balance precision and recall, since the default 0.5 threshold produces too many false positives on this heavily imbalanced data.



\## API Endpoints



\- `POST /predict` — submit a transaction, get a fraud prediction with SHAP explanation

\- `GET /transactions` — retrieve recent prediction history



Interactive API docs available at `/docs` when running locally.



\## Running Locally



```bash

\# Clone the repo

git clone https://github.com/Anuradhaprasad2007/fraud-detection.git

cd fraud-detection



\# Set up environment

python -m venv venv

venv\\Scripts\\Activate.ps1          # Windows

pip install -r requirements.txt



\# Add your database connection in a .env file

echo DATABASE\_URL=your\_postgres\_connection\_string > .env



\# Run the backend

python -m uvicorn src.main:app --reload



\# In a separate terminal, run the dashboard

streamlit run src\\dashboard.py

```



\## Project Structure



fraud-detection/

├── notebooks/ # EDA and model training notebooks

├── src/

│ ├── main.py # FastAPI app: /predict, /transactions

│ ├── database.py # Database connection setup

│ ├── models\_db.py # SQLAlchemy table schema

│ ├── dashboard.py # Streamlit frontend

│ └── models/

│ └── fraud\_xgboost\_final.pkl

├── requirements.txt

└── README.md





\## Dataset



\[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — a synthetic dataset simulating mobile money transactions, generated from real transaction logs. \~6.3M transactions, 0.13% labeled as fraud, occurring only in `CASH\_OUT` and `TRANSFER` transaction types.



\## Future Improvements



\- Retrain excluding post-transaction balance fields for a more conservative, deployment-realistic benchmark

\- Add automated tests and CI pipeline

\- Deploy backend and dashboard for public access



\## Team



Built by Anuradha Prasad (\[@Anuradhaprasad2007](https://github.com/Anuradhaprasad2007)), Nidhi Yadav (\[@NidhiYadav118](https://github.com/NidhiYadav118)) and Komal Kumari (\[@Komal385](https://github.com/komal385)) final-year BCA project.

