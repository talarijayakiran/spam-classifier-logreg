\# 📧 SMS Spam Classifier — Logistic Regression + FastAPI  

End-to-end machine learning project: training → evaluation → deployed inference API.





\##  Overview



This project builds a \*\*Spam Detection Model\*\* using Logistic Regression and TF-IDF text features.  

The trained model is deployed using \*\*FastAPI\*\*, providing a production-style real-time inference endpoint.



This project demonstrates real ML engineering skills:

\- Clean preprocessing pipeline  

\- TF-IDF feature engineering  

\- Saved ML pipeline (`joblib`)  

\- FastAPI inference service  

\- Professional project structure  



---



\##  Problem Statement



Spam messages are one of the most common communication threats.  

This project classifies SMS messages into:



\- \*\*ham (0)\*\* → legitimate  

\- \*\*spam (1)\*\* → unwanted / promotional  



---



\## 📂 Project Structure



```

spam-classifier-logreg/

│

├── app/                     # FastAPI application

│   └── main.py

│

├── notebooks/               # Jupyter training notebook

│   └── 01\_training.ipynb

│

├── models/                  # Model artifacts (ignored in Git)

│

├── data/                    # Dataset (ignored in Git)

│

├── requirements.txt

├── .gitignore

└── README.md

```



\##  Model Training (Summary)



\- TF-IDF used for text vectorization  

\- Logistic Regression as classifier  

\- Metrics evaluated:

&nbsp; - Accuracy  

&nbsp; - Precision  

&nbsp; - Recall  

&nbsp; - F1 Score  

&nbsp; - Confusion Matrix  



Model + preprocessing pipeline saved as:



```

models/spam\_pipeline.joblib

```



---



\## 🌐 FastAPI Inference Service



\### \*\*Run the API locally\*\*



```bash

uvicorn app.main:app --reload --port 8000

```



Swagger docs available at:



👉 http://127.0.0.1:8000/docs  

👉 http://127.0.0.1:8000/health



---



\## 📬 Example Prediction



\### Request:

```json

{

&nbsp; "text": "Congratulations! You won a free prize. Claim now!"

}

```



\### Response:

```json

{

&nbsp; "prediction": 1,

&nbsp; "probability": 0.9821,

&nbsp; "model\_version": "v1.0"

}

```



---



\## 🧱 Tech Stack



\- Python  

\- Scikit-Learn  

\- NLTK  

\- TF-IDF Vectorizer  

\- Logistic Regression  

\- FastAPI  

\- Uvicorn  



---



\## 🚀 Next Up (Future Improvements)



\- Add Docker container for deployment  

\- Add Streamlit UI for demo users  

\- Add unit tests + CI workflow  

\- Model versioning + monitoring  

\- REST + async optimizations  



---



\## 📄 License



This project is open-source under the MIT License.



---



\## 👤 Author

\*\*Talari Jaya Kiran\*\*  

MLOps / Machine Learning Engineer  



