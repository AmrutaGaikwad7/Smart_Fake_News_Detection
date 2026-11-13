# Fake News Detection Using NLP

A machine learning project that classifies news articles as **Fake** or **Real** using TF-IDF text features and two classification algorithms: Logistic Regression and Multinomial Naive Bayes. The trained models are evaluated and made available through an interactive Streamlit web application.

## 1. Project Overview



The project uses Natural Language Processing (NLP) and supervised machine learning to identify patterns in news articles and predict whether an article is Fake or Real.

## 2. Objectives

- Build a machine learning system for fake and real news classification.
- Perform text preprocessing and exploratory data analysis.
- Convert text into numerical features using TF-IDF.
- Train and compare Logistic Regression and Multinomial Naive Bayes.
- Evaluate models using standard classification metrics.
- Develop an interactive Streamlit application.
- Save and load trained models using Joblib.

## 3. Dataset

The project uses the **ISOT Fake News Dataset**, containing two CSV files:

- `Fake.csv` - Fake news articles.
- `True.csv` - Real news articles.

The dataset contains article titles, text, subjects and dates.

| Dataset information | Count |
|---|---:|
| Total original articles | 44,898 |
| Fake articles | 23,481 |
| Real articles | 21,417 |
| Cleaned and deduplicated articles used for training | 38,824 |

The original dataset contains 44,898 articles. After text cleaning and duplicate removal, 38,824 articles were retained for model development.

See `data/README.md` for dataset download and source information. Dataset usage is subject to the original dataset's terms.

## 4. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- NLTK / Python text processing
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## 5. Project Structure

## 5. Project Structure

```text
fake-news-detector/
|
|-- app.py
|-- README.md
|-- REPORT.md
|-- requirements.txt
|
|-- data/
|   |-- Fake.csv
|   |-- True.csv
|   |-- README.md
|
|-- src/
|   |-- __init__.py
|   |-- preprocess.py
|   |-- train.py
|   |-- eda.py
|
|-- scripts/
|   |-- make_sample_data.py
|   |-- build_report.py
|
|-- models/
|   |-- tfidf_vectorizer.joblib
|   |-- logistic_regression.joblib
|   |-- multinomial_nb.joblib
|   |-- best_model.joblib
|   |-- metrics.json
|
|-- reports/
|   |-- model_comparison.csv
|   |-- confusion_matrices.png
|   |-- roc_curves.png
|   |-- top_features.csv
|   |-- Fake_News_Detection_Project_Report.pdf
|
|-- outputs/
|   |-- visualizations/
|       |-- label_distribution.png
|       |-- article_length_distribution.png
|       |-- common_words_fake.png
|       |-- common_words_real.png
|       |-- dataset_inspection.csv
|
|-- screenshots/
    |-- Project screenshots
```

## 6. Data Preprocessing

The preprocessing pipeline includes:

1. Combining article titles and article text.
2. Converting text into a consistent format.
3. Removing unnecessary characters and noise.
4. Cleaning the text using the shared preprocessing function.
5. Removing empty and duplicate cleaned articles.
6. Assigning labels: Fake = 1 and Real = 0.

A stratified 80:20 train-test split is used with random state 42.

## 7. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset and identify patterns.

The following visualizations were generated:

- Fake vs Real news distribution.
- Article length distribution.
- Common words in Fake news.
- Common words in Real news.

The EDA output files are available in `outputs/visualizations/`.

## 8. Feature Extraction

**TF-IDF (Term Frequency-Inverse Document Frequency)** is used to convert news articles into numerical feature vectors.

Configuration includes:

- English stop-word removal.
- Unigrams and bigrams.
- Maximum features: 50,000.
- Minimum document frequency: 2.
- Maximum document frequency: 0.9.
- Sublinear term frequency scaling.

The vectorizer is fitted only on the training data to prevent data leakage.

## 9. Machine Learning Algorithms

Two classification algorithms were trained and compared:

**1. Logistic Regression**

A supervised classification algorithm that learns the relationship between TF-IDF features and news labels.

**2. Multinomial Naive Bayes**

A probabilistic classification algorithm commonly used for text classification tasks.

## 10. Model Training and Evaluation

The cleaned dataset was divided into training and testing sets.

| Training information | Value |
|---|---:|
| Training samples | 31,059 |
| Testing samples | 7,765 |
| TF-IDF vocabulary size | 50,000 |
| Best model | Logistic Regression |

The models were evaluated using accuracy, precision, recall, F1-score, ROC-AUC and cross-validation F1-score.

## 11. Model Comparison

Actual test-set results:

| Metric | Logistic Regression | Multinomial Naive Bayes |
|---|---:|---:|
| Accuracy | 0.9898 | 0.9598 |
| Precision | 0.9935 | 0.9569 |
| Recall | 0.9844 | 0.9559 |
| F1-score | 0.9889 | 0.9564 |
| ROC-AUC | 0.9992 | 0.9916 |
| Cross-validation F1 | 0.9891 | 0.9563 |

**Best Model: Logistic Regression**

Logistic Regression achieved the highest test F1-score and ROC-AUC among the two evaluated models.

These results reflect performance on the available test dataset and do not guarantee accuracy on every real-world news source.

## 12. Evaluation Visualizations

The project generates:

- Confusion matrices for both models.
- ROC curves.
- Model comparison results.
- Top features associated with Fake and Real predictions.

These files are saved in the `reports/` directory.

## 13. Streamlit Application

An interactive web application was developed using Streamlit.

Application features:

- News article text input.
- Fake or Real prediction.
- Prediction explanation using influential words.
- Model selection.
- Model evaluation dashboard.
- EDA and evaluation visualizations.

## 14. Model Persistence

Joblib is used to save the trained TF-IDF vectorizer and classification models.

Saved artifacts include:

- `tfidf_vectorizer.joblib`
- `logistic_regression.joblib`
- `multinomial_nb.joblib`
- `best_model.joblib`
- `metrics.json`

The application loads these saved artifacts instead of training the models again every time it starts.

## 15. Installation and Execution

### Step 1: Clone the repository

```powershell
git clone https://github.com/AmrutaGaikwad7/Smart_Fake_News_Detection.git
cd Smart_Fake_News_Detection
```

### Step 2: Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 4: Prepare the dataset

Place `Fake.csv` and `True.csv` inside the `data/` directory. Refer to `data/README.md` for download instructions.

### Step 5: Train the models

```powershell
python -m src.train
```

### Step 6: Generate EDA visualizations

```powershell
python -m src.eda
```

### Step 7: Run the Streamlit application

```powershell
python -m streamlit run app.py
```

The application will open in your web browser.

### Step 8: Generate the PDF project report

Install ReportLab if required:

```powershell
python -m pip install reportlab
```

Generate the report:

```powershell
python scripts/build_report.py
```

The PDF is generated at:

`reports/Fake_News_Detection_Project_Report.pdf`

## 16. Screenshots

The `screenshots/` directory contains evidence of:

1. Dataset overview and label distribution.
2. Article length and common-word EDA visualizations.
3. Model evaluation metrics.
4. Confusion matrices and model comparison.
5. Streamlit application interface.
6. Sample Fake/Real prediction.
7. Empty input validation.

## 17. Limitations and Future Scope

### Limitations

- The model learns patterns from its training dataset and may not generalize to every news source.
- New topics and changing writing styles may affect prediction performance.
- A machine learning prediction is not a substitute for independent fact-checking.
- Dataset bias may affect model predictions.

### Future Scope

- Add larger and more diverse news datasets.
- Explore advanced NLP models such as LSTM, BERT and transformer-based architectures.
- Improve generalization to new news sources.
- Add multilingual news classification.
- Integrate external fact-checking sources.

## 18. Conclusion

The project demonstrates how NLP and machine learning can be used to classify news articles as Fake or Real.

Both Logistic Regression and Multinomial Naive Bayes were trained and evaluated using TF-IDF features. Logistic Regression achieved the best overall test performance.

An interactive Streamlit application was developed to make the trained model accessible for predictions, and Joblib was used to persist the trained models and vectorizer.

## References

- ISOT Fake News Dataset - dataset source and download information are documented in `data/README.md`.
- Scikit-learn documentation: https://scikit-learn.org/
- Streamlit documentation: https://docs.streamlit.io/
- Pandas documentation: https://pandas.pydata.org/docs/
- Joblib documentation: https://joblib.readthedocs.io/
- Matplotlib documentation: https://matplotlib.org/stable/

