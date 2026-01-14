import random
import string


def random_id(n: int = 6) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))
