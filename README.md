# 🏥 Patient Feedback Analysis System

> **Comprehensive NLP-powered healthcare service insights from multilingual patient reviews**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Performance Indicators](#key-performance-indicators)
- [Methodology](#methodology)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)

---

## 🎯 Overview

This project implements an **end-to-end patient feedback analysis pipeline** that processes multilingual healthcare reviews to extract actionable insights. The system uses state-of-the-art NLP techniques including:

- **Multilingual embeddings** (sentence-transformers)
- **Advanced clustering** (K-Means + HDBSCAN)
- **Sentiment analysis** (transformer-based)
- **LLM-powered summarization** (Mistral-7B)
- **Interactive dashboard** (Streamlit)

### Dataset

- **Source**: German 2021 Patient Reviews (Kaggle)
- **Sample Size**: 6,000 reviews (random_state=42)
- **Languages**: German (25%), English (25%), French (25%), Arabic (25%)
- **Translation**: Automated using deep-translator

---

## ✨ Features

### 1. **Multilingual Support**
- Processes reviews in 4 languages (DE, EN, FR, AR)
- Language-specific stopword removal
- Unified embedding space for cross-lingual analysis

### 2. **Advanced NLP Pipeline**
- **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
- **Clustering**: Optimal K selection via Silhouette Score
- **Topic Modeling**: BERTopic with keyword extraction
- **Sentiment**: XLM-RoBERTa-based classification

### 3. **Automated Insights**
- Top 10 recurring issues identified
- Cluster-wise formal summaries (clinical tone)
- Business impact scoring
- Sentiment-topic correlation analysis

### 4. **Recommendation Engine**
- Rule-based + ML-driven recommendations
- Severity classification (Critical/High/Medium/Low)
- Category mapping (Operational, Staff, Clinical, etc.)
- Evidence-backed action items

### 5. **Interactive Dashboard**
- Real-time filtering (language, sentiment, cluster, severity)
- UMAP visualization of embeddings
- Sentiment trends and distributions
- Exportable insights and reports

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Raw Data       │
│  (Kaggle)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  - Sampling     │
│  - Cleaning     │
│  - Translation  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embedding      │
│  (Multilingual  │
│   Transformer)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Clustering     │
│  - K-Means      │
│  - HDBSCAN      │
│  - UMAP         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  NLP Analysis   │
│  - Sentiment    │
│  - Keywords     │
│  - Topics       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Summary    │
│  (Mistral-7B)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Recommendations│
│  & Dashboard    │
└─────────────────┘
```

---

## 🚀 Installation & Running

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for LLM)
- 16GB+ RAM

### Local Setup & Running

```bash
git clone <repository-url>
cd patient-feedback-analysis

python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt

streamlit run streamlit_app.py
```

### Google Colab Setup

```python
# Upload notebooks to Colab
# Run in order: 01 → 02 → 03 → 04 → 05

# Each notebook has installation cells
# GPU Runtime recommended (Runtime → Change runtime type → T4 GPU)
```

## 📖 Usage

### Step-by-Step Execution

#### 1. Data Exploration
```bash
jupyter notebook notebooks/01_setup_and_exploration.ipynb
```
- Loads and explores raw data
- Identifies text/rating/date columns
- Creates initial visualizations

#### 2. Preprocessing & Translation
```bash
jupyter notebook notebooks/02_preprocessing_translation.ipynb
```
- Samples 6,000 reviews (random_state=42)
- Cleans and anonymizes text
- Translates to 4 languages
- Removes stopwords (multilingual)

#### 3. NLP Analysis
```bash
jupyter notebook notebooks/03_embedding_and_nlp_analysis.ipynb
```
- Creates sentence embeddings
- Performs UMAP dimensionality reduction
- Clusters using K-Means + HDBSCAN
- Analyzes sentiment (XLM-RoBERTa)
- Extracts keywords per cluster

#### 4. LLM Summarization
```bash
jupyter notebook notebooks/04_llm_summarization.ipynb
```
- Loads Mistral-7B-Instruct (4-bit quantized)
- Generates formal summaries for each cluster
- Identifies top 10 recurring issues
- Calculates business impact scores

#### 5. Recommendations
```bash
jupyter notebook notebooks/05_recommendation_engine.ipynb
```
- Classifies issues into categories
- Calculates severity levels
- Generates actionable recommendations
- Creates detailed justifications for top 3

#### 6. Launch Dashboard
```bash
streamlit run streamlit_app.py
```
- Open browser to http://localhost:8501
- Explore interactive visualizations
- Filter by language/sentiment/cluster
- Export insights

---

## 📁 Project Structure

```
patient-feedback-analysis/
│
├── data/
│   ├── raw/                      # Original dataset
│   ├── processed/                # Cleaned & translated data
│   └── embeddings/               # Sentence embeddings & UMAP
│
├── notebooks/
│   ├── 01_setup_and_exploration.ipynb
│   ├── 02_preprocessing_translation.ipynb
│   ├── 03_embedding_and_nlp_analysis.ipynb
│   ├── 04_llm_summarization.ipynb
│   └── 05_recommendation_engine.ipynb
│
├── outputs/
│   ├── figures/                  # All visualizations
│   ├── recommendations/          # Generated recommendations
│   ├── cluster_insights_with_summaries.csv
│   ├── top_10_recurring_issues.csv
│   └── executive_summary.json
│
├── models/
│   └── cache/                    # Saved models
│
├── streamlit_app.py              # Dashboard application
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
├── README.md                     # This file
└── architecture.md               # Detailed architecture docs
```

---


## 🔬 Methodology

### 1. Data Preprocessing
- **Sampling**: Stratified random sampling (n=6000, seed=42)
- **Cleaning**: URL/email removal, whitespace normalization
- **Anonymization**: Phone number/ID masking
- **Translation**: Google Translator via deep-translator
- **Stopwords**: NLTK multilingual + custom healthcare terms

### 2. Embedding Generation
- **Model**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **Normalization**: L2 normalization for cosine similarity
- **Batch Size**: 32 (GPU optimized)

### 3. Clustering
- **Method**: K-Means (primary), HDBSCAN (validation)
- **Optimal K**: Silhouette Score + Davies-Bouldin Index
- **Dimensionality Reduction**: UMAP (n_neighbors=15, min_dist=0.1)

### 4. Sentiment Analysis
- **Model**: cardiffnlp/twitter-xlm-roberta-base-sentiment
- **Classes**: Negative, Neutral, Positive
- **Validation**: Cross-referenced with rating scores

### 5. Summarization
- **Model**: Mistral-7B-Instruct-v0.2 (4-bit quantized)
- **Prompt Template**: Formal hospital report style
- **Output Format**: Exactly 2 sentences, clinical tone
- **Temperature**: 0.3 (consistent, factual output)

### 6. Recommendation Generation
- **Classification**: Rule-based keyword matching
- **Categories**: Operational, Staff, Clinical, Administrative, etc.
- **Severity**: Multi-factor scoring (sentiment + volume + rating)
- **Actions**: Template-based with context adaptation

---


## 🚀 Future Enhancements

### Short-term (Stretch Goals)

#### 1. **Multi-language Enhancements**
- [ ] Translate keywords and needed data to English to have an integrated English dashboard (It needed local GPU for translation & I did not have)

#### 2. **Live Data Integration**
- [ ] API integration for real-time review ingestion (It was a very easy step but again I could not implement it because of GPU)
- [ ] Scheduled data refresh (daily/weekly)

#### 3. **Advanced LLM Features**
- [ ] Tone adaptation (clinician vs administrator summaries)
- [ ] Multi-stakeholder report generation
- [ ] Automated email summaries

#### 4. **Predictive Analytics**
-  My dataset did not have any date column so implementing predictive analytics was not possible.

### Long-term Roadmap

#### 1. **Model Improvements**
- Fine-tune sentiment model on healthcare data
- Train custom topic model for medical domain
- Implement aspect-based sentiment analysis

#### 2. **Dashboard Enhancements**
- Real-time monitoring mode
- Comparative analysis (hospital-to-hospital)
- Mobile-responsive design
- PDF report generation

#### 4. **Advanced Analytics**
- Patient journey analysis
- Cohort segmentation
- Causal inference modeling
- A/B testing framework for interventions

---

## 🛠️ Technical Stack

### Core Technologies
- **Python 3.8+**: Primary language
- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace models
- **Streamlit**: Dashboard framework

### NLP Libraries
- **sentence-transformers**: Multilingual embeddings
- **NLTK**: Text preprocessing
- **spaCy**: Advanced NLP tasks
- **BERTopic**: Topic modeling

### ML/Data Science
- **scikit-learn**: Clustering, metrics
- **UMAP**: Dimensionality reduction
- **HDBSCAN**: Density-based clustering
- **pandas/numpy**: Data manipulation

### Visualization
- **Plotly**: Interactive charts
- **Matplotlib/Seaborn**: Static plots
- **WordCloud**: Keyword visualization

### Translation
- **deep-translator**: Multi-provider translation
- **langdetect**: Language identification

---


## 🔄 Version History

- **v1.0.0** (2025-01-XX): Initial release
  - Multilingual support (4 languages)
  - 6 clustering algorithms tested
  - LLM-powered summarization
  - Interactive dashboard
  - Recommendation engine

---

**Built with ❤️ for healthcare service improvement**