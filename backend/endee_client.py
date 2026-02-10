import numpy as np


class EndeeClient:
    """
    Simulated Endee Vector Database Client.

    This class mimics how a real vector database (like Endee)
    would store and search embeddings.

    In production:
    - Embeddings would be stored inside Endee service
    - Search would be executed via Endee API
    """

    def __init__(self):
        self.vectors = []
        self.texts = []

    def add_document(self, embedding, text):
        """
        Store embedding and corresponding text.
        """
        self.vectors.append(np.array(embedding))
        self.texts.append(text)

    def search(self, query_embedding, top_k=3):
        """
        Perform cosine similarity search.
        Returns top_k most similar text chunks.
        """

        if not self.vectors:
            return ["No documents indexed yet. Please call /index first."]

        query_vector = np.array(query_embedding)

        similarities = []

        for vector in self.vectors:
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            similarities.append(similarity)

        # Get indices of top_k highest similarity scores
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append(self.texts[idx])

        return results
