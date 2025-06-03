# INFO284 Final Exam – Machine Learning Projects

This repository contains the final exam solutions for INFO284, covering two comprehensive machine learning tasks:
1. **Sentiment Analysis** of hotel reviews.
2. **Image Classification** focused on frog detection from the CIFAR-10 dataset.

---

## 📊 Task I: Sentiment Analysis

**Goal:** Classify hotel reviews as either *bad* (score ≤ 8) or *good* (score > 8).

### Methodology
- **Data Source:** Hotel_Reviews.csv
- **Preprocessing:**
  - Merged positive and negative reviews
  - Cleaned text: lowercasing, punctuation removal, stopword filtering (kept "not", "no", "never"), lemmatization
- **Feature Extraction:**
  - TF-IDF with unigrams and bigrams (10,000 max features)
- **Models Trained:**
  - Logistic Regression
  - Naive Bayes
  - Decision Tree
  - LSTM (with Keras)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score

### Results Summary
| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.81     | 0.76      | 0.67   | 0.71     |
| Naive Bayes         | 0.79     | 0.74      | 0.64   | 0.69     |
| Decision Tree       | 0.72     | 0.65      | 0.45   | 0.53     |
| LSTM                | 0.77     | 0.65      | 0.78   | 0.71     |

**Conclusion:** Logistic Regression showed the best balance of performance and interpretability.

---

## 🖼️ Task II: Image Classification (Frog Detector)

**Goal:** Detect whether an image contains a frog (binary classification based on CIFAR-10).

### Methodology
- **Dataset:** CIFAR-10 (via torchvision)
- **Preprocessing:**
  - Normalization and tensor conversion
  - Augmentation on frog class: crop, flip, jitter
- **Model:** Custom CNN
  - 2 convolutional layers with batch normalization, ReLU, max pooling
  - 3 fully connected layers with dropout
  - Final output: 2 logits (frog / not-frog)
- **Training:**
  - Optimizer: AdamW
  - Loss: CrossEntropy
  - EarlyStopping + ModelCheckpoint

### Results Summary
| Class     | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Non-frog | 0.97      | 0.98   | 0.97     |
| Frog     | 0.77      | 0.73   | 0.75     |

**Conclusion:** CNN achieved strong overall accuracy (~95%). Recall for frog class was reasonable given class imbalance.

---

## 🛠 Technologies Used
- Python
- Pandas, NumPy
- scikit-learn
- NLTK
- TensorFlow / Keras
- PyTorch / torchvision
- PyTorch Lightning
- Matplotlib, Seaborn

---

## 📁 Project Structure
