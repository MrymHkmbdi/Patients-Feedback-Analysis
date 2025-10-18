**Solution Design Document (SDD)**
==================================

### *Patient Feedback Analysis System*

**Multilingual NLP for Healthcare Service Insights**

* * * * *

**1\. Purpose & Business Context**
----------------------------------

The *Patient Feedback Analysis System* enables healthcare leaders to transform multilingual patient reviews into measurable, actionable service insights.

The goal is to **standardize patient experience analytics** across languages and facilities, allowing executives to:

-   Quantify patient satisfaction and pain points,

-   Detect recurring operational and clinical issues,

-   Prioritize interventions by severity and business impact, and

-   Track improvement over time.

Without a structured analytical design, the organization risks inconsistent reporting, missed trends, and limited ability to act on patient voice data.

* * * * *

**2\. Stakeholder Requirements → Analytical Variables**
-------------------------------------------------------

| **Business Question** | **Analytical Translation** | **Measurement Output** |
| --- | --- | --- |
| 1\. What are the top recurring complaints? | NLP clustering on multilingual embeddings | Cluster IDs + issue categories |
| 2\. How severe is each issue? | Multi-factor severity formula (sentiment × volume × rating gap) | Severity score (0--100) |
| 3\. How many patients are affected? | Cluster size / total sample | % Share of total |
| 4\. What's driving dissatisfaction? | Keyword extraction post-stopword removal | Top 15 keywords per cluster |
| 5\. What actions should we take? | Rule-based recommendation mapping | Prioritized action list |
| 6\. Is the model trustworthy? | Sentiment--rating correlation validation | Pearson r ≥ 0.75 |
| 7\. Do insights vary by language? | Cross-language cluster overlap | % Shared topics |
| 8\. How can leadership act quickly? | Streamlit interactive dashboard | 5-min executive summary view |

Each requirement is traceable to a defined **data field or metric** in the analytics layer.

* * * * *

**3\. Data Foundation**
-----------------------

| **Attribute** | **Description** | **Source/Transformation** |
| --- | --- | --- |
| `review_text` | Patient feedback (multilingual) | Kaggle DE 2021 dataset, sampled (6,000 reviews) |
| `rating_normalized` | Rating scaled 1--5 | Derived from raw rating |
| `target_language` | Language label | Langdetect auto-classification |
| `sentiment` | NLP sentiment (neg/neu/pos) | XLM-RoBERTa sentiment model |
| `cluster_kmeans` | Cluster assignment | K-Means on UMAP-5D embeddings |
| `keywords` | Top terms per cluster | TF frequency filtering |
| `severity_score` | Weighted metric (see formula) | Derived measure |
| `recommended_action` | Human-interpretable action | Rule-based generator |
| `impact_score` | Combined KPI for ranking | Composite metric |

These variables form the **reporting layer** used by the Streamlit dashboard and all leadership metrics.

* * * * *

**4\. KPI & Metric Design**
---------------------------

| **KPI** | **Formula / Definition** | **Purpose** |
| --- | --- | --- |
| **Sample Coverage** | processed_reviews ÷ 6,000 | Pipeline completeness |
| **Negative Sentiment %** | (negative_reviews ÷ total_reviews) × 100 | Emotional risk indicator |
| **Avg Rating (×1.2)** | mean(rating_normalized) × 1.2 | Business-normalized satisfaction |
| **Severity Score** | 0.4×neg% + 0.3×(cluster%×10) + 0.3×((5--avg_rating)×20) | Priority weighting |
| **Top-5 Coverage** | sum(top5_cluster_sizes) ÷ total | Concentration index |
| **Recommendation Coverage** | % clusters with actionable recs | Readiness for intervention |

Severity classification:

-   **Critical ≥70** → Immediate corrective action

-   **High 50--69** → 30-day improvement target

-   **Medium 30--49** → Quarterly optimization

-   **Low <30** → Monitor only

* * * * *

**5\. Solution Architecture**
-----------------------------

**End-to-End Workflow**

1.  **Ingestion & Preprocessing**

    -   Import Kaggle dataset → sample 6K records

    -   Clean, anonymize, detect language, translate (DE→EN/FR/AR)

    -   Remove stopwords (NLTK + medical lexicon)

2.  **Embedding & Clustering**

    -   Model: `paraphrase-multilingual-MiniLM-L12-v2`

    -   Dimensionality: UMAP (5D for clustering; 2D for visualization)

    -   Clustering: K-Means (optimal K via Silhouette)

    -   Validation: HDBSCAN

3.  **Sentiment & Keyword Analysis**

    -   Model: `cardiffnlp/twitter-xlm-roberta-base-sentiment`

    -   Keyword extraction via token frequency and exclusion list

4.  **Summarization & Recommendation Engine**

    -   LLM: Mistral-7B-Instruct (quantized for GPU efficiency)

    -   Generate summaries per cluster

    -   Classify issues → operational categories

    -   Attach ranked, evidence-backed recommendations

5.  **Dashboard & Delivery (Streamlit)**

    -   Tabs: Overview | Cluster Details

    -   Metrics: total reviews, negative%, avg rating

    -   Visuals: cluster size bar, rating bars, impact map, UMAP visualization

    -   Detail view: cluster metrics, sentiment mix, recommended actions, sample reviews



* * * * *

**6\. Success Metrics & Future Scope**
--------------------------------------

✅ Executives can identify top 10 pain points and corresponding actions in **<5 minutes**.\
✅ Dashboard loads in **<5 seconds** and supports real-time filters.\
✅ All critical issues have **quantified severity** and **recommended remediation**.\
✅ Pipeline completes 6K reviews in **<2 hours** on a standard GPU.
