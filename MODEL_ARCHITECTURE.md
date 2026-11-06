# MED-OS // Clinical AI Diagnostic Model Architecture & Inner Workings

This document provides a publication-grade breakdown of the mathematical foundations, machine learning pipeline, feature engineering, and clinical triage heuristics powering the **MED-OS Clinical Diagnosis Engine**.

---

## 1. System Overview & Objective

The objective of the MED-OS Diagnostic Model is to ingest unstructured patient-reported symptoms, extract clinical linguistic tokens, map them into a high-dimensional vector space, and resolve a probabilistic diagnosis across 15 distinct medical disease classes.

```
+--------------------------------------------------------------------------------------------------------+
|                                  MED-OS CLINICAL DIAGNOSTIC PIPELINE                                   |
+--------------------------------------------------------------------------------------------------------+
|                                                                                                        |
|  [ Patient Input Text ]                                                                                |
|          |                                                                                             |
|          v                                                                                             |
|  [ Stage 1: Regex Lexical Sanitization & Normalization ] --> Strip non-alphanumeric, lowercase         |
|          |                                                                                             |
|          v                                                                                             |
|  [ Stage 2: NLTK Stopword Pruning & Unigram/Bigram Tokenizer ] --> Remove noise words                 |
|          |                                                                                             |
|          v                                                                                             |
|  [ Stage 3: Sublinear TF-IDF Feature Projection (639-D) ] --> Sparse vectorization                     |
|          |                                                                                             |
|          v                                                                                             |
|  [ Stage 4: Multinomial Logistic Discriminant Engine ] --> L2 Regularization, L-BFGS Hyperplanes       |
|          |                                                                                             |
|          v                                                                                             |
|  [ Stage 5: Softmax Probability Calibration & Clinical Triage ] --> Urgency, Specialist & Protocol     |
|                                                                                                        |
+--------------------------------------------------------------------------------------------------------+
```

---

## 2. Dataset Anatomy

The training corpus (`comprehensive_medical_dataset.csv`) contains **450 verified clinical records** meticulously balanced across 15 diseases:

| Disease Class | Sample Count | Primary Biomarkers | Urgency Tier | Typical Specialist |
|---|---|---|---|---|
| **Common Cold** | 30 | sneezing, runny nose, nasal congestion, mild cough | Routine | General Physician |
| **Influenza** | 30 | chills, high fever, myalgia, fatigue, severe cough | Urgent | General Physician / Pulmonologist |
| **COVID-19** | 30 | anosmia, dyspnea, fever, dry cough, throat irritation | Urgent | Pulmonologist / Infectious Disease |
| **Hypertension** | 30 | dizziness, occipital headache, palpitations, chest pressure | Routine / 24-hr | Cardiologist |
| **Diabetes Type 2** | 30 | polydipsia, polyuria, blurred vision, chronic fatigue | Routine | Endocrinologist |
| **Heart Disease** | 30 | substernal chest tightness, radiating arm pain, diaphoresis | Urgent (ER) | Cardiologist |
| **Pneumonia** | 30 | productive cough, pleuritic pain, high fever, dyspnea | Urgent | Pulmonologist |
| **Migraine** | 30 | unilateral throbbing pain, photophobia, phonophobia, aura | High Priority | Neurologist |
| **Asthma** | 30 | expiratory wheezing, bronchoconstriction, dyspnea | High Priority | Pulmonologist |
| **Depression** | 30 | anhedonia, persistent low mood, insomnia, psychomotor slowing | Routine | Psychiatrist |
| **Anxiety Disorder** | 30 | restlessness, tachycardia, muscle tension, hypervigilance | Routine | Psychiatrist / Clinical Psychologist |
| **Arthritis** | 30 | joint inflammation, stiffness, reduced range of motion | Routine | Rheumatologist |
| **Gastritis** | 30 | epigastric burning, postprandial nausea, acid reflux | Routine | Gastroenterologist |
| **UTI** | 30 | dysuria, urinary frequency, suprapubic cramping | 24-Hour | Urologist / Nephrologist |
| **Kidney Stones** | 30 | acute flank colic radiating to groin, hematuria, nausea | Urgent | Urologist / Nephrologist |

---

## 3. Mathematical Foundations

### 3.1 Term Frequency - Inverse Document Frequency (TF-IDF)
For any term $t$ in patient input $d$ relative to corpus $D$:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \left( \ln\frac{1 + |D|}{1 + \text{DF}(t, D)} + 1 \right)$$

Where:
- $\text{TF}(t, d)$ represents the sublinear term frequency within the query.
- $\text{DF}(t, D)$ is the document frequency of term $t$ across all 450 clinical records.
- Unigram and bigram extraction ($n \in \{1, 2\}$) captures both individual symptoms (`wheezing`) and clinical compound markers (`chest tightness`).
- Active vocabulary space: **639 dimensions**.

### 3.2 Multinomial Logistic Regression & Softmax Formulation
The classifier computes 15 linear discriminant functions corresponding to each disease $c \in \{1, \dots, 15\}$:

$$z_c = \mathbf{w}_c^T \mathbf{x} + b_c$$

Where $\mathbf{w}_c \in \mathbb{R}^{639}$ is the weight vector for disease $c$, $\mathbf{x}$ is the sparse TF-IDF input vector, and $b_c$ is the class bias.

The class probabilities are derived using the Softmax function:

$$P(y = c \mid \mathbf{x}) = \frac{\exp(z_c)}{\sum_{k=1}^{15} \exp(z_k)}$$

### 3.3 Regularization & Optimization Objective
The weights $\mathbf{W}$ and biases $\mathbf{b}$ are trained by minimizing the L2-regularized multinomial cross-entropy loss:

$$\min_{\mathbf{W}, \mathbf{b}} -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K \mathbb{I}(y_i = k) \ln P(y = k \mid \mathbf{x}_i) + \frac{1}{2C} \sum_{k=1}^K \|\mathbf{w}_k\|_2^2$$

The optimization is solved iteratively using the **L-BFGS** (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) quasi-Newton algorithm to ensure smooth global convergence.

---

## 4. Local Explainability & Feature Attribution

Unlike black-box neural networks, MED-OS provides **exact linear feature attribution**:

$$\text{Attribution}(t_j) = w_{pred, j} \times x_j$$

- If $\text{Attribution}(t_j) > 0$: The word actively pushed the model's confidence towards the diagnosed condition.
- If $\text{Attribution}(t_j) < 0$: The word pushed the model away from this condition (differentiating it from competing diagnoses).

---

## 5. Clinical Triage Heuristics

Once the predicted disease $c^*$ and probability $P(y = c^* \mid \mathbf{x})$ are computed:
1. **Urgency Classification**:
   - `CRITICAL`: Immediate emergency evaluation required (e.g. Heart Disease, acute dyspnea).
   - `WITHIN 24 HOURS`: Prompt primary or specialist evaluation needed (e.g. UTI, acute severe migraine).
   - `ROUTINE`: Outpatient consultation advised (e.g. Arthritis, Diabetes Type 2).
2. **Confidence Thresholding**:
   - If $P < 0.65$: Low statistical confidence warning is appended, prompting the user that non-specific multi-etiological symptoms require laboratory panels.

