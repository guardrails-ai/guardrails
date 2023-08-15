def mock_create_embedding(input, *args, **kwargs):
    mocked_embeddings = {
        "It was a beautiful day. " "In the afternoon, we drank tea.": [0, 0.5],
        "Then we went to the park. "
        "There was a lot of people there. "
        "A dog was there too.": [0.5, 0],
        "It was a nice day.": [0.25, 0.25],
        "I went to the park.": [0.25, 0.25],
        "I saw a dog.": [0.25, 0.25],
    }

    if not isinstance(input, list):
        input = [input]

    returns = []
    for text in input:
        try:
            returns.append({"embedding": mocked_embeddings[text]})
        except KeyError:
            print(input)
            raise ValueError("Text not found in mocked embeddings")
    return {"data": returns}
