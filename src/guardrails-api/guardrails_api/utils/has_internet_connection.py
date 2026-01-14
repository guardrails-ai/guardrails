import requests


def has_internet_connection() -> bool:
    try:
        res = requests.get("https://www.guardrailsai.com/")
        res.raise_for_status()
        return True
    except requests.ConnectionError:
        return False
