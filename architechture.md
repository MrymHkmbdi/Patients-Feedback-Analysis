# System Architecture - Patient Feedback Analysis

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │  Raw Data    │───▶│  Processed   │───▶│  Embeddings  │    │
│  │  (Kaggle)    │    │  Data        │    │  (Vectors)   │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│         │                    │                    │            │
└─────────┼────────────────────┼────────────────────┼────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐│
│  │Preprocessing│  │ Translation│  │ Embedding  │  │Clustering││
│  │  Module    │─▶│   Module   │─▶│   Module   │─▶│  Module  ││
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘│
│                                                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐              │
│  │ Sentiment  │  │  Keyword   │  │    LLM     │              │
│  │  Analysis  │  │ Extraction │  │Summarizer  │              │
│  └────────────┘  └────────────┘  └────────────┘              │
│         │                │                │                    │
└─────────┼────────────────┼────────────────┼────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYTICS LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐         ┌─────────────────────┐         │
│  │  Recommendation  │         │   Insight Generator │         │
│  │     Engine       │◀───────▶│    (Metrics & KPIs) │         │
│  └──────────────────┘         └─────────────────────┘         │
│           │                              │                     │
└───────────┼──────────────────────────────┼─────────────────────┘
            │                              │
            ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PRESENTATION LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│          ┌────────────────────────────────┐                    │
│          │   Streamlit Dashboard          │                    │
│          │  - Interactive Filters         │                    │
│          │  - Real-time Visualizations    │                    │
│          │  - Data Export                 │                    │
│          └────────────────────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Component Details

### 1. Data Layer

#### 1.1 Raw Data Module
- **Input**: Kaggle dataset (German patient reviews)
- **Format**: CSV with text, ratings, dates
- **Storage**: `data/raw/`

#### 1.2 Processed Data Module
- **Functions**:
  - Sampling (n=6000, random_state=42)
  - Cleaning (URL/email removal, whitespace)
  - Anonymization (phone/ID masking)
  - Language distribution (25% per language)
- **Output**: `data/processed/processed_multilingual.csv`
- **Schema**:
  ```
  - text_original: str (German)
  - text_cleaned: str (cleaned German)
  - text_final: str (translated)
  - text_no_stopwords: str (processed)
  - target_language: str (de/en/fr/ar)
  - rating_normalized: float (1-5)
  ```

#### 1.3 Embeddings Module
- **Model**: paraphrase-multilingual-MiniLM-L12-v2
- **Dimension**: 384
- **Storage**: `data/embeddings/sentence_embeddings.npy`
- **Format**: numpy array (6000, 384)
- **Additional**: UMAP projections (2D, 5D)

---

### 2. Processing Layer

#### 2.1 Preprocessing Module
**Components**:
- Text Cleaner
  - Regex-based pattern matching
  - Special character handling
  - Preserve medical terminology
  
- Anonymizer
  - Phone number detection: `\+?\d[\d\s\-\(\)]{7,}\d`
  - ID pattern removal: `\b[A-Z0-9]{5,}\b`
  
- Language Detector
  - Library: langdetect
  - Confidence threshold: 0.8

**Flow**:
```
Raw Text → Clean → Anonymize → Detect Language → Store
```

#### 2.2 Translation Module
**Strategy**:
- Provider: Google Translator (via deep-translator)
- Batch processing: 1 text per request
- Rate limiting: 0.1s delay between calls
- Error handling: Retry logic (3 attempts)
- Fallback: Keep original on failure

**Language Distribution**:
```python
total_samples = 6000
samples_per_lang = 1500

German (de):  samples[0:1500]      # Original
English (en): samples[1500:3000]   # Translated from DE
French (fr):  samples[3000:4500]   # Translated from DE
Arabic (ar):  samples[4500:6000]   # Translated from DE
```

#### 2.3 Embedding Module
**Model Architecture**:
- Base: sentence-transformers
- Model: paraphrase-multilingual-MiniLM-L12-v2
- Training: 50+ languages, 1B+ sentence pairs
- Pooling: Mean pooling
- Normalization: L2

**Parameters**:
```python
batch_size = 32
max_seq_length = 512
normalize_embeddings = True
device = "cuda" if available else "cpu"
```


#### 2.4 Clustering Module
**Methods Implemented**:

1. **K-Means**
   - Algorithm: Lloyd's algorithm
   - Initialization: k-means++
   - n_init: 20 (stability)
   - Metric: Euclidean distance on UMAP-5D
   
2. **HDBSCAN**
   - min_cluster_size: 50
   - min_samples: 10
   - metric: Euclidean
   - Selection: Excess of Mass (EOM)

**Optimal K Selection**:
```python
k_range = [5, 6, 7, ..., 20]

for k in k_range:
    silhouette_score = calculate_silhouette(k)
    davies_bouldin_score = calculate_db(k)

optimal_k = argmax(silhouette_score)
```

#### 2.5 Sentiment Analysis Module
**Model**:
- Name: cardiffnlp/twitter-xlm-roberta-base-sentiment
- Base: XLM-RoBERTa (multilingual)
- Classes: Negative, Neutral, Positive
- Max length: 512 tokens

**Pipeline**:
```python
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model_name,
    device=0 if cuda else -1,
    batch_size=16
)
```


#### 2.6 Keyword Extraction Module
**Method**: Frequency-based + filtering

```python
def extract_keywords(texts, top_n=10):
    # Tokenize
    words = tokenize(texts)
    
    # Filter
    words = [w for w in words if:
        len(w) > 3 and          # Minimum length
        not w.isdigit() and     # Not numeric
        w not in stopwords      # Not stopword
    ]
    
    # Count and rank
    return Counter(words).most_common(top_n)
```

**Stopwords**:
- Standard: NLTK lists (DE, EN, FR, AR)
- Custom: Medical terms, common names
- Total: ~500-700 per language

#### 2.7 LLM Summarization Module
**Model Setup**:
```python
model = "mistralai/Mistral-7B-Instruct-v0.2"
quantization = "4-bit"  # BitsAndBytes
compute_dtype = torch.float16
device_map = "auto"
```

**Prompt Engineering**:
```
System: You are writing an internal hospital service report.

Instructions:
You are writing an internal hospital service report.
Summarize the following cluster of patient comments in exactly 2 plain sentences.
Sentence 1 must start with this exact phrase:
'Main issues of the patients in this topic were '
and then list recurring issues or observations (general, impersonal).
Sentence 2 should state overall sentiment or operational implication.
Use a neutral, factual, clinical tone.
Do not reference people, journalists, organizations, articles, locations, or outside events.
Do not use pronouns, names, dates, or greetings.
Do not invent information. Avoid introductions like 'Welcome', 'Overall', or 'In summary'.

Input: [Cluster texts]
Output: [2-sentence summary]
```

**Generation Parameters**:
```python
temperature = 0.3      # Low for consistency
top_p = 0.9          
max_new_tokens = 200
repetition_penalty = 1.15
```

---

### 3. Analytics Layer

#### 3.1 Recommendation Engine

**Architecture**:
```
Input: Cluster insights
  ↓
Category Classification (Rule-based)
  ↓
Severity Scoring (Multi-factor)
  ↓
Action Mapping (Template-based)
  ↓
Evidence Collection
  ↓
Output: Ranked recommendations
```

**Category Classification**:
```python
categories = {
    'waiting': ['wart', 'wait', 'delay', ...],
    'staff': ['freundlich', 'friendly', ...],
    'communication': ['erkl', 'explain', ...],
    'billing': ['rechnun', 'billing', ...],
    # ... more categories
}

def classify(keywords, summary):
    scores = {cat: count_matches(cat, keywords) 
              for cat in categories}
    return max(scores, key=scores.get)
```

**Severity Calculation**:
```python
severity_score = (
    negative_sentiment% * 0.4 +
    cluster_size% * 0.3 +
    rating_gap * 0.3
)

if severity_score >= 70: return "Critical"
if severity_score >= 50: return "High"
if severity_score >= 30: return "Medium"
return "Low"
```



---
## 🔄 Data Flow Diagram

```
┌──────────┐
│ Raw CSV  │
└────┬─────┘
     │
     ▼
┌─────────────────┐
│ Load & Sample   │ (random_state=42, n=6000)
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│ Clean & Anon    │ (regex, masking)
└────┬────────────┘
     │
     ├────────────────────┬────────────────────┬────────────────────┐
     ▼                    ▼                    ▼                    ▼
┌─────────┐         ┌─────────┐         ┌─────────┐         ┌─────────┐
│Keep DE  │         │Trans EN │         │Trans FR │         │Trans AR │
│(1500)   │         │(1500)   │         │(1500)   │         │(1500)   │
└────┬────┘         └────┬────┘         └────┬────┘         └────┬────┘
     │                   │                   │                   │
     └───────────────────┴───────────────────┴───────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Remove Stopwords│
                        └────┬────────────┘
                             │
                             ▼
                        ┌─────────────────┐
                        │ Create Embeddings│ (sentence-transformers)
                        └────┬────────────┘
                             │
                             ├────────────────────┐
                             ▼                    ▼
                        ┌─────────┐         ┌─────────┐
                        │ UMAP    │         │ Cluster │
                        │ 2D/5D   │         │ K-Means │
                        └─────────┘         └────┬────┘
                                                  │
                        ┌─────────────────────────┴────────┐
                        │                                  │
                        ▼                                  ▼
                 ┌─────────────┐                  ┌──────────────┐
                 │  Sentiment  │                  │   Keywords   │
                 │   Analysis  │                  │  Extraction  │
                 └──────┬──────┘                  └──────┬───────┘
                        │                                │
                        └────────────┬───────────────────┘
                                     ▼
                            ┌─────────────────┐
                            │ LLM Summarize   │ (Mistral-7B)
                            └────┬────────────┘
                                 │
                                 ▼
                            ┌──────────────────┐
                            │  Recommendations │
                            └────┬─────────────┘
                                 │
                                 ▼
                            ┌──────────────────┐
                            │    Dashboard     │
                            └──────────────────┘
```

---

## 🔐 Security & Privacy

### Data Protection
- **Anonymization**: Automatic PII masking
- **Storage**: Local filesystem (no cloud by default)
- **Access**: No authentication (add as needed)

### Model Security
- **Local execution**: No API calls (except translation)
- **Audit**: All processing logged

---

**Document Version**: 1.0
**Last Updated**: 2025-10
**Maintained By**: Maryam Hokmabadi