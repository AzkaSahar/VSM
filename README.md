# Vector Space Model (VSM) – Information Retrieval System

## Objective

This assignment implements a **Vector Space Model (VSM)** for information retrieval.

* Build a document-term vector space using **TF-IDF** weighting.
* Preprocess a collection of **448 computer science abstracts**.
* Compute **cosine similarity** between query and document vectors.
* Rank and filter documents based on similarity threshold.
* An **intuitive GUI** to visualize and search through the document set.

---

## Files

* `Abstracts` — Folder containing 448 `.txt` files (English abstracts).
* `Stopword-List.txt` — List of stopwords to remove during preprocessing.
* `Gold_Standard.txt` — Evaluation set of expected results (15 example queries).

---

## Features Implemented

### Preprocessing

* **Tokenization**, **case folding**, **punctuation stripping**.
* **Stopword removal** using the provided stopword list.
* **Lemmatization** using POS tags via NLTK.
* **Hyphenated word splitting** and **alphanumeric extraction**.

### Indexing

* **Inverted Index**: Maps terms to document IDs.
* **Positional Index**: Maps terms to their positions within documents.

### TF-IDF Computation

* `TF = 1 + log10(tf)` if `tf > 0`.
* `IDF = log(N / df)` where `N` is total documents, `df` is document frequency.
* Weights stored and reused via a JSON file (`indexes.json`).

### Vector Space & Ranking

* Each document and query represented as a vector in `ℝⁿ`.
* Cosine similarity between query vector and each document vector.
* Documents are ranked and filtered using a similarity threshold (e.g., `α = 0.05`).

### GUI 

Built using `customtkinter` with the following:

* Fullscreen UI with escape-to-exit shortcut.
* Query input + search button.
* Result display for top-ranked document.
* 2D visualization of document embeddings using **t-SNE** (bonus visualization).
---

## How to Run

### Requirements

Install Python dependencies:

```bash
pip install nltk scikit-learn customtkinter matplotlib chardet
```

Also, download NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```
Run the main file:

```bash
python final.py
```

Use the **GUI** to input queries and explore document matches.

---

## Example Query Usage

Type a query like:

```
machine learning algorithms
```

Top results will be displayed in the GUI and plotted in 2D space.

https://github.com/user-attachments/assets/e081b5b0-6e29-4d50-8aa0-f61b0f4f0aaa
