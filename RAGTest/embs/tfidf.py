from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TFIDFEmbedding:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit(self, documents):
        """
        Fit the TF-IDF vectorizer on the provided documents.

        Args:
            documents (list of str): List of text documents to fit the vectorizer.
        """
        self.vectorizer.fit(documents)

    def transform(self, documents):
        """
        Transform the documents into TF-IDF vectors.

        Args:
            documents (list of str): List of text documents to transform.

        Returns:
            np.ndarray: TF-IDF vectors for the documents.
        """
        return self.vectorizer.transform(documents).toarray()

    def fit_transform(self, documents):
        """
        Fit the vectorizer and transform the documents in one step.

        Args:
            documents (list of str): List of text documents to fit and transform.

        Returns:
            np.ndarray: TF-IDF vectors for the documents.
        """
        return self.vectorizer.fit_transform(documents).toarray()

    def get_feature_names(self):
        """
        Get the feature names (vocabulary) learned by the vectorizer.

        Returns:
            list of str: Feature names.
        """
        return self.vectorizer.get_feature_names_out()

# Example usage
if __name__ == "__main__":
    documents = ["This is a sample document.", "This document is another example."]
    tfidf = TFIDFEmbedding()
    tfidf.fit(documents)
    vectors = tfidf.transform(documents)
    print("Feature Names:", tfidf.get_feature_names())
    print("TF-IDF Vectors:", vectors)