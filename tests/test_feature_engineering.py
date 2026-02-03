import pytest
import numpy as np

from clustering_pipeline.feature_engineering.feature_engineering import process_images_features


class FakeEmbeddingModel:
    def generate_embedding(self, text: str):
        n = len(text)

        return np.array([
            float(n),
            float(n + 1),
            float(n + 2)
        ])


@pytest.fixture
def embedding_model():
    return FakeEmbeddingModel()


@pytest.fixture
def sample_images():

    return {
        "img1": {
            "doc_type": "invoice",
            "issuer": "ACME",
            "amounts": [100, 200],
            "dates": ["2023-01-01", "2023-02-01"],
            "language": "en"
        },
        "img2": {
            "doc_type": "receipt",
            "issuer": "SHOP",
            "amounts": [50],
            "dates": ["2023-01-15"],
            "language": "fr"
        }
    }


@pytest.fixture
def weights():

    return {
        "doc": 1.0,
        "issuer": 1.0,
        "amount": 1.0,
        "date": 1.0,
        "lang": 1.0
    }


def test_numeric_scaled(
    sample_images,
    weights,
    embedding_model
):

    out = process_images_features(
        sample_images,
        weights,
        embedding_model
    )

    amounts = np.array([
        out["img1"]["amount_scaled"],
        out["img2"]["amount_scaled"]
    ])

    dates = np.array([
        out["img1"]["date_scaled"],
        out["img2"]["date_scaled"]
    ])

    # mean approx 0
    assert np.isclose(amounts.mean(), 0, atol=1e-6)
    assert np.isclose(dates.mean(), 0, atol=1e-6)

    # std approx 1
    assert np.isclose(amounts.std(), 1, atol=1e-6)
    assert np.isclose(dates.std(), 1, atol=1e-6)


def test_missing_fields(
    weights,
    embedding_model
):

    data = {
        "img": {}
    }

    out = process_images_features(
        data,
        weights,
        embedding_model
    )

    v = out["img"]["vector"]

    assert len(v) > 0


def test_length_of_fused_vector(
    sample_images,
    weights,
    embedding_model
):

    out = process_images_features(
        sample_images,
        weights,
        embedding_model
    )

    v1 = out["img1"]["vector"]
    v2 = out["img2"]["vector"]

    assert len(v1) == len(v2) == 10 # 3 (doc emb) + 3 (issuer emb) + 1 (amount) + 1 (date) + 2 (lang)

