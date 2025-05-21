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
                print(f"[Query TF-IDF] Term: '{word}', TF: {tf:.3f}, IDF: {idf:.3f}, TF-IDF: {tfidf_weight:.5f}")
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
            print(f"[CosSim] Doc: {doc_id}, Dot: {dot_product:.5f}, |Q|: {query_norm:.5f}, |D|: {doc_norm:.5f}, Sim: {sim:.6f}")

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