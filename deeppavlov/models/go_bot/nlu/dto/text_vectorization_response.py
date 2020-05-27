class TextVectorizationResponse:
    """
    Stores the BOW-encodings and (padded or aggregated e.g. averaged) embeddings for text.
    """

    def __init__(self, tokens_bow_encoded, tokens_aggregated_embedding, tokens_embeddings_padded):
        self.tokens_bow_encoded = tokens_bow_encoded
        self.tokens_aggregated_embedding = tokens_aggregated_embedding
        self.tokens_embeddings_padded = tokens_embeddings_padded
