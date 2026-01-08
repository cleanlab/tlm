import random

import numpy as np


def fake_embedding(content: str, length: int) -> list[float]:
    """Fake embedding for a given content.

    This function is deterministic, but it does not have the property that
    strings that are close in semantic distance are close in vector distance.

    Returns a unit vector of the given length, computed deterministically based
    on content.
    """
    # Initialize a random number generator seeded with the content
    # to ensure that the same content always generates the same vector
    #
    # This is not a CSPRNG, but that is fine for our purposes
    rng = random.Random(content)

    # Generate a vector of random floats, with each element in [0, 1)
    vector = [rng.random() for _ in range(length)]

    # Calculate the magnitude of the vector
    magnitude = sum(x**2 for x in vector) ** 0.5

    # Normalize the vector to unit length
    #
    # This vector is not a uniform random unit vector, but that is fine for our
    # purposes
    return [x / magnitude for x in vector]


def fake_embedding_with_target_cosine_distance(orig_embedding: list[float], target_distance: float) -> list[float]:
    orig = np.array(orig_embedding)
    orig = orig / np.linalg.norm(orig)

    # Create a random vector orthogonal to orig
    rand = np.random.randn(*orig.shape)
    rand -= np.dot(rand, orig) * orig  # make orthogonal
    rand /= np.linalg.norm(rand)

    # Compute angle theta from cosine similarity
    target_cosine = 1 - target_distance
    theta = np.arccos(target_cosine)

    # Combine original and orthogonal vector to get new vector
    new: list[float] = (np.cos(theta) * orig + np.sin(theta) * rand).tolist()
    return new
