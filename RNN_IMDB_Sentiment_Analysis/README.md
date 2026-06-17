# RNN IMDB Sentiment Analysis

## Overview

This project implements a Recurrent Neural Network (RNN) using PyTorch to perform sentiment analysis on movie reviews from the IMDB dataset.

The model predicts whether a review expresses a positive or negative sentiment after applying several Natural Language Processing (NLP) preprocessing techniques.

The project was built to gain practical experience with:

- Natural Language Processing (NLP)
- Text Cleaning and Preprocessing
- TF-IDF Vectorization
- Recurrent Neural Networks (RNN)
- Binary Text Classification
- PyTorch Deep Learning Workflow

---

## Problem Statement

Given a movie review, predict whether the sentiment is:

- Positive
- Negative

This is a binary classification problem.

---

## Dataset

### IMDB Movie Reviews Dataset

The dataset contains movie reviews along with their sentiment labels.

Target Classes:

| Label | Meaning |
|---------|----------|
| 0 | Negative |
| 1 | Positive |

---

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- NLTK
- Scikit-Learn

---

## Text Preprocessing Pipeline

The following preprocessing steps were performed before model training:

### 1. Convert Text to Lowercase

```text
I LOVE THIS MOVIE
↓
i love this movie
```

### 2. Remove URLs

```text
https://example.com
```

removed using regular expressions.

### 3. Remove HTML Tags

```html
<p>This movie is great</p>
```

↓

```text
This movie is great
```

### 4. Remove Punctuation

```text
Amazing!!!
```

↓

```text
Amazing
```

### 5. Remove Stopwords

Examples:

```text
the, is, are, was, and
```

### 6. Stemming

Examples:

```text
running → run
playing → play
studies → studi
```

using Porter Stemmer.

### 7. Label Encoding

Sentiment labels were converted into numerical values.

---

## Feature Engineering

### TF-IDF Vectorization

Text reviews were transformed into numerical feature vectors using:

```python
TfidfVectorizer(max_features=5000)
```

This converts textual data into machine-readable numerical representations.

Maximum Vocabulary Size:

```text
5000 Features
```

---

## Train-Test Split

Dataset was split into:

```text
80% Training
20% Testing
```

using:

```python
train_test_split()
```

---

## Data Preparation

The TF-IDF sparse matrix was converted into:

```python
PyTorch Tensors
```

and loaded using:

```python
TensorDataset
DataLoader
```

Batch Size:

```text
64
```

---

## RNN Architecture

### Network Structure

```text
Input Features
(5000 TF-IDF Features)

↓

Simple RNN Layer
Hidden Size = 128

↓

Fully Connected Layer

↓

Sigmoid Activation

↓

Sentiment Prediction
```

---

## Model Configuration

### Hidden Size

```text
128
```

### Number of Layers

```text
1
```

### Loss Function

```python
Binary Cross Entropy Loss (BCELoss)
```

### Optimizer

```python
Adam Optimizer
```

### Epochs

```text
10
```

---

## Training Workflow

1. Load and preprocess reviews
2. Apply TF-IDF vectorization
3. Convert features to tensors
4. Create DataLoaders
5. Forward propagation
6. Compute binary cross entropy loss
7. Backpropagation
8. Update weights using Adam optimizer
9. Evaluate on test dataset

---

## Model Evaluation

The trained model was evaluated using:

### Classification Accuracy

```python
correct_predictions / total_predictions
```

Predictions were generated using:

```python
torch.sigmoid()
```

and classified as:

```text
Probability > 0.5 → Positive
Probability ≤ 0.5 → Negative
```

---

## Project Structure

```text
RNN_IMDB_Sentiment_Analysis/
│
├── RNN_IMDB_Sentiment_Analysis.ipynb
└── README.md
```

---

## Key Concepts Practiced

- Natural Language Processing
- Text Cleaning
- Regular Expressions
- Stopword Removal
- Stemming
- TF-IDF Vectorization
- Binary Classification
- Recurrent Neural Networks
- PyTorch DataLoaders
- Binary Cross Entropy Loss
- Model Evaluation

---

## Future Improvements

- Replace TF-IDF with Word Embeddings
- Use LSTM Networks
- Use GRU Networks
- Add Attention Mechanism
- Use Pretrained Embeddings (GloVe, Word2Vec)
- Hyperparameter Tuning
- Use Transformer-Based Models (BERT)

---

## Author

Altaf Jawed

Artificial Intelligence • Deep Learning • Natural Language Processing
