# Email Spam Classification Using Machine Learning and LoRA Fine-Tuning


## Overview
This project compares traditional machine learning models and transformer based language models for email spam classification.

Five models were evaluated:

- MLP Baseline (TF-IDF features)
- MLP Enhanced (deeper neural network + regularization)
- DistilBERT Full Fine-Tuning
- LoRA Fine-Tuning (r=4)
- LoRA Fine-Tuning (r=16)

The goal is to analyze the performance difference between classic ML approaches and parameter-efficient fine-tuning methods.

## Dataset & Preprocessing
- Dataset: Kaggle Email Spam Classification Dataset
- Original samples: 10,000 emails
- Removed duplicates: 8,223
- Final dataset: 1,777 unique emails
  - Ham: 1,000
  - Spam: 777

Preprocessing steps:
- Text cleaning
- Stop word removal
- TF-IDF feature extraction
- Stratified train/test split
- Label noise simulation for training robustness

## Results

| Model | Accuracy | F1 Score |
|------|----------|----------|
| MLP Baseline | 0.900 | 0.881 |
| MLP Enhanced | 0.978 | 0.974 |
| DistilBERT | 0.990 | 0.988 |
| LoRA r=4 | 0.988 | 0.986 |
| LoRA r=16 | 0.989 | 0.987 |

## Key Findings
- Transformer models achieved the highest performance by understanding contextual language patterns.
- LoRA fine-tuning achieved almost the same performance as full fine-tuning while training only a small percentage of parameters.
- LoRA r=16 provided the best efficiency/performance tradeoff.

## Technologies Used
- Python
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)

## Future Work
- Test on larger real-world email datasets
- Explore larger transformer models
- Deploy the model as an API
- Evaluate real-time inference performance
