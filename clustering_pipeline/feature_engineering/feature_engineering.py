import numpy as np
from datetime import datetime
from typing import Dict
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    normalize
)
from .embedding_llm.embedding_llm import EmbeddingLLM

from .embedding_llm.embedding_llm_openai import OpenAIEmbeddingLLM


def process_images_features(
    images_features: Dict[str, Dict],
    feature_weights: Dict[str, float],
    embedding_model: EmbeddingLLM = OpenAIEmbeddingLLM(),
) -> Dict[str, Dict]:
    """
    Process features and builds fused feature vectors for clustering.
    """

    records = []


    # phase 1: extract raw features

    for image_id, features in images_features.items():

        # text embeddings
        doc_type = features.get("doc_type", "")
        issuer = features.get("issuer", "")

        doc_emb = embedding_model.generate_embedding(doc_type)
        issuer_emb = embedding_model.generate_embedding(issuer)

        # amount (log-transformed max)
        amounts = features.get("amounts", [])
        max_amount = max(amounts) if amounts else 0.0
        log_amount = np.log1p(max_amount)

        # date (latest → timestamp)
        dates = features.get("dates", [])
        if dates:
            latest = max(dates)
            ts = datetime.fromisoformat(latest).timestamp()
        else:
            ts = 0.0

        # language
        lang = features.get("language", "unknown")

        records.append({
            "id": image_id,
            "doc_emb": doc_emb,
            "issuer_emb": issuer_emb,
            "amount": log_amount,
            "date": ts,
            "lang": lang
        })


    # phase 2: encode & normalize

    # normalize embeddings
    doc_embs = normalize(np.array([r["doc_emb"] for r in records]))
    issuer_embs = normalize(np.array([r["issuer_emb"] for r in records]))

    # scale numeric features
    numeric = np.array([
        [r["amount"], r["date"]]
        for r in records
    ])
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric)
    amount_scaled = numeric_scaled[:, 0].reshape(-1, 1)
    date_scaled = numeric_scaled[:, 1].reshape(-1, 1)

    # one-hot encode language
    langs = [[r["lang"]] for r in records]
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )
    lang_oh = encoder.fit_transform(langs)


    # phase 3: weighted fusion

    X = np.hstack([
        feature_weights.get("doc",      1.0) * doc_embs,
        feature_weights.get("issuer",   1.0) * issuer_embs,
        feature_weights.get("amount",   1.0) * amount_scaled,
        feature_weights.get("date",     1.0) * date_scaled,
        feature_weights.get("lang",     1.0) * lang_oh
    ])


    # phase 4: build output dict

    output = {}
    for i, r in enumerate(records):
        output[r["id"]] = {
            "vector": X[i].tolist(),
            "doc_type_embedding": doc_embs[i].tolist(),
            "issuer_embedding": issuer_embs[i].tolist(),
            "amount_scaled": amount_scaled[i, 0].tolist(),
            "date_scaled": date_scaled[i, 0].tolist(),
            "language_onehot": lang_oh[i].tolist()
        }

    return output
