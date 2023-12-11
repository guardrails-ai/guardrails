from typing import List

GOOD_TRANSLATION = "This is a good translation."
BAD_TRANSLATION = "This is a bad translation."


class MockModel:
    def predict(self, data: list, **kwargs):
        return MockOutput(data)


class MockOutput:
    scores: List[float]

    def __init__(self, data: list):
        # return a score of 0.9 for good translation and 0.4 for bad translation
        data = data[0]
        if data["mt"] == GOOD_TRANSLATION:
            self.scores = [0.9]
        elif data["mt"] == BAD_TRANSLATION:
            self.scores = [0.4]
