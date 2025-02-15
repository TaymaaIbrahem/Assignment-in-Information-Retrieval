# Information Retrieval Course - Practical Assignment

## Authors
- **Tayma Ibrahem**  
- **Oron Kam**  

## Overview
This project was conducted as part of the *Information Retrieval* course and explores various IR-related tasks. It focuses on:
- **Language Modeling** using a unigram model with smoothing techniques.
- **Text Classification** using multiple machine learning classifiers.
- **Text Clustering** using vector space models and k-means clustering.

The project was developed in **Python** and implemented in **Google Colab**, utilizing libraries such as **NLTK, Scikit-learn, and PCA**.

---
## Task 1: Language Modeling

We built a **unigram language model** using **Simple Good-Turing smoothing** to estimate word probabilities. The workflow involved:

### Steps:
1. **Data Preprocessing:**
   - Read text files from the "Train and Test" directories.
   - Tokenize the text using NLTK.
   - Evaluate four different linguistic preprocessing techniques:
     1. No preprocessing
     2. Case folding
     3. Stemming (Porter Algorithm)
     4. Stop-word removal
2. **Dataset Splitting:**
   - 80% Training
   - 10% Testing
   - 10% Validation
3. **Model Training:**
   - Train a **unigram model** using the **Simple Good-Turing** method to estimate word probabilities.
4. **Model Evaluation:**
   - Measure **Perplexity** to evaluate the model's effectiveness.

### Results:
| Model Type              | Test Perplexity | Validation Perplexity |
|-------------------------|----------------|----------------------|
| No Preprocessing       | 834.79         | 841.84               |
| Case Folding           | 94.13          | 91.55                |
| Stemming               | 24.22          | 24.30                |
| Stop-Word Removal      | 19.29          | 19.62                |

**Key Takeaways:**
- **Stop-word removal** had the most significant impact on reducing perplexity.
- **Stemming and case folding** also improved performance but to a lesser extent.

---
## Task 2: Text Classification

We performed text classification using **four different models**:
- **Logistic Regression**
- **Naïve Bayes**
- **Support Vector Machine (SVM)**
- **Random Forest**

### Workflow:
1. **Preprocessing:** Load documents and split into **90% train, 10% test**.
2. **Feature Extraction:** Convert text into a **vector space model (TF-IDF)**.
3. **Model Training & Evaluation:**
   - Metrics used: **Accuracy, Precision, Recall**

### Results:
| Model               | Accuracy | Precision | Recall  |
|---------------------|----------|-----------|---------|
| Logistic Regression | 1.0000   | 1.0000    | 1.0000  |
| Naïve Bayes        | 1.0000   | 1.0000    | 1.0000  |
| SVM                | 0.5455   | 0.7980    | 0.5455  |
| Random Forest      | 0.9091   | 0.9273    | 0.9091  |

**Key Observations:**
- **Logistic Regression & Naïve Bayes** achieved perfect classification scores.
- **Random Forest** performed well but was not perfect.
- **SVM** struggled due to **lower recall**, making it less suitable for this dataset.

---
## Task 3: Text Clustering

We performed document clustering using **k-means** and evaluated performance using:
- **Adjusted Rand Index (ARI)**
- **Silhouette Score**

### Workflow:
1. **Text Vectorization:** Convert documents into **TF-IDF feature vectors**.
2. **Dimensionality Reduction:** Apply **PCA** for visualization.
3. **Clustering:** Use **k-means** to group similar documents.
4. **Evaluation:**
   - Compute **ARI** to measure similarity between true and predicted clusters.
   - Compute **Silhouette Score** to assess clustering quality.

### Results:
| Metric | Score  |
|--------|--------|
| Adjusted Rand Index (ARI) | 0.6691 |
| Silhouette Score          | 0.666  |

**Observations:**
- The data was **naturally distributed into 3 clusters**, but we initially used **k=4**.
- Running k-means again with **k=3** would likely yield improved results.
- The presence of **non-globular clusters** impacted k-means' performance.

---
## Conclusion
- **Language modeling:** Stop-word removal significantly improved model efficiency.
- **Text classification:** Logistic Regression and Naïve Bayes were the best models.
- **Clustering:** K-means worked well but required optimal **k selection**.

This project provided valuable hands-on experience with **language modeling, classification, and clustering in Information Retrieval.**

---
## Technologies Used
- **Programming Language:** Python
- **Libraries:** NLTK, Scikit-learn, Matplotlib, Pandas, NumPy
- **Development Environment:** Google Colab

---
## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   ```
2. Open **Google Colab** and upload the notebook.
3. Install dependencies:
   ```python
   !pip install nltk scikit-learn pandas matplotlib
   ```
4. Run the notebook cell by cell.

---
## Future Improvements
- Optimize **k-means clustering** with better **feature selection**.
- Extend **language modeling** to **bigram/trigram models**.
- Experiment with **deep learning classifiers** (e.g., LSTMs, transformers).

---
## Acknowledgments
This project was completed as part of the *Information Retrieval* course and guided by our instructors.

---
## License
This project is licensed under the **MIT License**. Feel free to use and modify the code.

---
