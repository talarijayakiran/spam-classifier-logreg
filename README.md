---

## 📘 How to Run This Project (Step-by-Step Instructions)

This section provides full instructions to reproduce the dataset preparation, model training, and deployment using FastAPI and Streamlit.

---

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/talarijayakiran/spam-classifier-logreg
cd spam-classifier-logreg
```

---

### 🔹 2. Create & Activate Virtual Environment (Windows)

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

---

### 🔹 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🔹 4. Dataset Setup

Download the **SMS Spam Collection Dataset** from:

UCI Repository:  
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

Place the downloaded CSV file here:

```
data/spam.csv
```

> Note: `data/` is ignored in GitHub to keep the repo clean.

---

### 🔹 5. Train the Model (Jupyter Notebook)

Open the training notebook:

```
notebooks/01_training.ipynb
```

Running all cells will:

- Load and clean the text data  
- Apply preprocessing (lowercase, punctuation removal, tokenization, stopwords)  
- Convert text to TF-IDF features  
- Train a Logistic Regression model  
- Evaluate the model  
- Save the final pipeline to:

```
models/spam_pipeline.joblib
```

---

### 🔹 6. Run FastAPI (Model Inference API)

Start the API server:

```bash
uvicorn app.main:app --reload --port 8000
```

Open in browser:

- Swagger UI → **http://127.0.0.1:8000/docs**
- Health check → **http://127.0.0.1:8000/health**

---

### 🔹 7. Run Streamlit Web App (Bonus UI)

```bash
streamlit run app/ui/streamlit_app.py
```

This launches an interactive interface where you can:

- Enter any SMS message  
- Get prediction (Spam / Ham)  
- View probability score  

---

## 📚 Data Source

**SMS Spam Collection Dataset**  
UCI Machine Learning Repository  
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

Classes:

- **0 = ham**  
- **1 = spam**

---

##  Libraries Used

- pandas  
- numpy  
- scikit-learn  
- nltk  
- joblib  
- fastapi  
- uvicorn  
- streamlit  

---

##  Evaluation Summary

| Metric        | Score     |
|---------------|-----------|
| Accuracy      | **0.9677** |
| Precision     | **0.9829** |
| Recall        | **0.7723** |
| F1 Score      | **0.8650** |

### Key Insight:
- Ham detection is nearly perfect.
- Spam detection is strong but can be improved by tuning TF-IDF features or using Naive Bayes/SVM.

---

## 📝 Conclusion

This project demonstrates a complete ML workflow:

- Data preprocessing  
- TF-IDF vectorization  
- Logistic Regression training  
- Evaluation  
- Saving and loading ML pipelines  
- Deployment using FastAPI  
- Streamlit UI for interactive classification  

It reflects real-world ML engineering and MLOps practices.

---
