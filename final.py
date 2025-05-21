import os
import json
import chardet
import numpy as np
import string
from tkinter import messagebox
from customtkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from math import log
from numpy.linalg import norm
import re
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, stopwords):
    text = text.replace("/", " ")
    tokens = word_tokenize(text.lower())
    pos_tags = pos_tag(tokens)

    processed = []
    for word, tag in pos_tags:
        raw_word = word.strip(string.punctuation)

        if not raw_word or raw_word in stopwords:
            continue

        pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(raw_word, pos)
        processed.append(lemma)

        # Handle hyphenated words
        if '-' in word:
            parts = word.split('-')
            for part in parts:
                part = part.strip(string.punctuation)
                if part and part not in stopwords:
                    processed.append(part)

        # Alphanumeric components
        extras = re.findall(r'[a-zA-Z0-9]+', word)
        for e in extras:
            if e != raw_word and e not in stopwords:
                processed.append(e)

    return processed

# === Data Handling ===

def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        return set(file.read().splitlines())

def read_documents(folder_path, stopwords):
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                encoding = chardet.detect(f.read())['encoding'] or 'utf-8'
            with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                words = preprocess_text(file.read(), stopwords)
                documents[filename.split(".")[0]] = words
    return documents

def build_inverted_index(documents):
    inverted_index = {}
    for doc_id, words in documents.items():
        for word in words:
            inverted_index.setdefault(word, set()).add(doc_id)
    return {word: sorted(docs, key=int) for word, docs in inverted_index.items()}

def build_positional_index(documents):
    positional_index = {}
    for doc_id, words in documents.items():
        for pos, word in enumerate(words):
            positional_index.setdefault(word, {}).setdefault(doc_id, []).append(pos)
    return positional_index

# Updated TF-IDF computation with 1 + log10(tf) if tf > 0
def compute_tfidf(documents, inverted_index):
    tfidf = {}
    N = len(documents)
    idf = {term: log(N / len(doc_ids)) if len(doc_ids) != 0 else 0 for term, doc_ids in inverted_index.items()}

    for doc_id, words in documents.items():
        tfidf[doc_id] = {}
        word_counts = {word: words.count(word) for word in set(words)}

        for word, count in word_counts.items():
            if count > 0:
                tf = 1 + log(count, 10)
                tfidf_weight = tf * idf.get(word, 0)
                tfidf[doc_id][word] = tfidf_weight
                print(f"[TF-IDF] Doc: {doc_id}, Term: '{word}', TF: {tf:.3f}, IDF: {idf.get(word, 0):.3f}, TF-IDF: {tfidf_weight:.5f}")

    return tfidf, idf


def save_indexes(inverted_index, positional_index, tfidf, idf, filename="indexes.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "inverted_index": inverted_index,
            "positional_index": positional_index,
            "tfidf": tfidf,
            "idf": idf
        }, f, indent=4)

def load_indexes(filename="indexes.json"):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data["inverted_index"], data["positional_index"], data["tfidf"], data.get("idf", {})
    return None, None, None, {}

# === GUI Application ===

class App(CTk):
    def __init__(self):
        super().__init__()
        self.title("VSM Information Retrieval System")
        self.geometry("800x600")
        self.attributes('-fullscreen', True)
        self.configure(bg="#0F1B2B")
        self.bind("<Escape>", self.toggle_fullscreen)

        self.sidebar_frame = CTkFrame(self, width=200, fg_color="#1C1C28", border_color="#14A098", border_width=2)
        self.sidebar_frame.pack(side="left", fill="y", pady=10)
        self.sidebar_frame.pack_propagate(False)

        self.sidebar_inner_frame = CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.sidebar_inner_frame.place(relx=0.5, rely=0, anchor="n", y=30)

        self.plot_frame = CTkFrame(self, fg_color="#0F1B2B")
        self.plot_frame.pack(expand=True, fill="both", pady=10)

        common_font = ("Montserrat", 14)

        button_style = {
            "fg_color": "#CB2D6F",
            "text_color": "#F0F0F0",
            "hover_color": "#E84575",
            "corner_radius": 50,
            "font": common_font,
        }
        input_style = {
            "fg_color": "#1C1C28",
            "text_color": "#F0F0F0",
            "corner_radius": 6,
            "font": common_font,
            "border_color": "#14A098",
            "border_width": 2
        }
        label_style = {
            "text_color": "#F0F0F0",
            "font": common_font
        }

        self.query_input = CTkEntry(self.sidebar_inner_frame, placeholder_text="Enter query...", width=160, **input_style)
        self.query_input.pack(pady=(30,10))

        self.search_button = CTkButton(self.sidebar_inner_frame, text="Search", command=self.handle_query, **button_style)
        self.search_button.pack(pady=5)

        self.result_text = CTkLabel(self.sidebar_inner_frame, text="Result: N/A", wraplength=160, justify="left", **label_style)
        self.result_text.pack(pady=(30, 10))

        self.exit_button = CTkButton(self.sidebar_frame, text="Exit", command=self.quit, **button_style)
        self.exit_button.place(relx=0.5, rely=1.0, anchor="s", y=-20)

        self.ax = None
        self.canvas = None
        self.stopwords = set()
        self.documents = {}
        self.inverted_index = {}
        self.positional_index = {}
        self.tfidf = {}
        self.idf = {}
        self.doc_ids = []
        self.doc_matrix = None
        self.doc_coords = None
        self.vocab = []
        self.term_to_index = {}
        self.initialize_indexing()

    def toggle_fullscreen(self, event=None):
        self.attributes('-fullscreen', not self.attributes('-fullscreen'))

    def initialize_indexing(self):
        try:
            self.stopwords = load_stopwords("Stopword-List.txt")
            self.inverted_index, self.positional_index, self.tfidf, self.idf = load_indexes()

            if not self.inverted_index or not self.positional_index or not self.tfidf or not self.idf:
                folder = os.path.join(os.getcwd(), "Abstracts")
                if not os.path.exists(folder):
                    messagebox.showerror("Folder Missing", "'Abstracts' folder not found.")
                    self.destroy()
                    return
                self.documents = read_documents(folder, self.stopwords)
                self.inverted_index = build_inverted_index(self.documents)
                self.positional_index = build_positional_index(self.documents)
                self.tfidf, self.idf = compute_tfidf(self.documents, self.inverted_index)
                save_indexes(self.inverted_index, self.positional_index, self.tfidf, self.idf)

            self.plot_document_vectors()
        except Exception as e:
            messagebox.showerror("Indexing Error", str(e))
            self.destroy()

    def plot_document_vectors(self):
        self.doc_ids = sorted(self.tfidf.keys(), key=lambda x: int(x))
        self.vocab = sorted({term for doc in self.tfidf.values() for term in doc})
        self.term_to_index = {term: i for i, term in enumerate(self.vocab)}

        self.doc_matrix = np.zeros((len(self.doc_ids), len(self.vocab)))
        for i, doc_id in enumerate(self.doc_ids):
            for term, score in self.tfidf[doc_id].items():
                j = self.term_to_index[term]
                self.doc_matrix[i, j] = score

        M_normalized = normalize(self.doc_matrix)
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
        self.doc_coords = tsne.fit_transform(M_normalized)

        self.draw_plot()

    def draw_plot(self, query_coords=None):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0F1B2B")
        ax.set_facecolor("#0F1B2B")

        x_coords, y_coords = self.doc_coords[:, 0], self.doc_coords[:, 1]
        ax.scatter(x_coords, y_coords, color="#CB2D6F", s=16)

        if query_coords is not None:
            ax.scatter(query_coords[0], query_coords[1], color="#14A098", s=100, label="Query", marker='*')
            ax.legend()

        ax.set_xlabel("X-axis", fontsize=12, fontname="Montserrat", color="#F0F0F0")
        ax.set_ylabel("Y-axis", fontsize=12, fontname="Montserrat", color="#F0F0F0")
        ax.set_title("Document Vectors (t-SNE)", fontsize=14, fontname="Montserrat", color="#F0F0F0")
        ax.tick_params(axis='x', colors='#F0F0F0')
        ax.tick_params(axis='y', colors='#F0F0F0')
        ax.axhline(0, color='#F0F0F0', linewidth=0.5, ls='--')
        ax.axvline(0, color='#F0F0F0', linewidth=0.5, ls='--')
        ax.grid(True)
        fig.tight_layout()

        self.ax = ax
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def handle_query(self):
        query = self.query_input.get().strip()
        if not query:
            self.result_text.configure(text="Result: Enter a query.")
            return

        tokens = preprocess_text(query, self.stopwords)
        word_counts = {word: tokens.count(word) for word in set(tokens)}
        query_vector = np.zeros(len(self.vocab))

        print(f"\n[Query] '{query}' Tokens: {tokens}")

        for word, count in word_counts.items():
            if word in self.term_to_index:
                tf = 1 + log(count, 10)
                idf = self.idf.get(word, 0)
                tfidf_weight = tf * idf
                query_vector[self.term_to_index[word]] = tfidf_weight
            else:
                print(f"[Query TF-IDF] Term: '{word}' not found in vocabulary.")

        similarities = {}
        query_norm = norm(query_vector)

        for i, doc_vector in enumerate(self.doc_matrix):
            doc_norm = norm(doc_vector)
            dot_product = np.dot(query_vector, doc_vector)

            if doc_norm == 0 or query_norm == 0:
                sim = 0.0
            else:
                sim = dot_product / (query_norm * doc_norm)

            doc_id = self.doc_ids[i]
            similarities[doc_id] = sim

        alpha = 0.001
        relevant = sorted([doc for doc, sim in similarities.items() if sim >= alpha], key=lambda x: int(x))

        if relevant:
            self.result_text.configure(text="Relevant: " + ", ".join(relevant[:200]) + ("..." if len(relevant) > 200 else ""))
        else:
            self.result_text.configure(text="No relevant documents found.")

        query_matrix = normalize(query_vector.reshape(1, -1))
        all_points = np.vstack([self.doc_matrix, query_vector])
        all_normalized = normalize(all_points)
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
        all_coords = tsne.fit_transform(all_normalized)
        query_coords = all_coords[-1]

        self.doc_coords = all_coords[:-1]
        self.draw_plot(query_coords=query_coords)


app = App()
app.mainloop()
