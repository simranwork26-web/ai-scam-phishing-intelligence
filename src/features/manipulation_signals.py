import re


PATTERNS = {
    "urgency": re.compile(
        r"\b(?:urgent|immediately|asap|action required|act now|"
        r"within \d+ (?:hour|hours|day|days)|deadline)\b",
        re.IGNORECASE,
    ),
    "financial": re.compile(
        r"\b(?:payment|pay|paid|invoice|refund|bank|account number|"
        r"credit card|debit card|wire transfer|transfer money)\b",
        re.IGNORECASE,
    ),
    "authority": re.compile(
        r"\b(?:IT support|helpdesk|administrator|admin|security team|"
        r"support team|HR department|bank|government|police)\b",
        re.IGNORECASE,
    ),
}


def detect_manipulation_signals(text: str) -> dict:
    text = str(text or "")

    matches = {
        name: bool(pattern.search(text))
        for name, pattern in PATTERNS.items()
    }

    signals = [
        name for name, detected in matches.items()
        if detected
    ]

    combinations = []

    if matches["urgency"] and matches["financial"]:
        combinations.append("urgency + financial")

    if matches["urgency"] and matches["authority"]:
        combinations.append("urgency + authority")

    if matches["financial"] and matches["authority"]:
        combinations.append("financial + authority")

    if (
        matches["urgency"]
        and matches["financial"]
        and matches["authority"]
    ):
        combinations.append("urgency + financial + authority")

    return {
        "individual_signals": signals,
        "combinations": combinations,
        "urgency": matches["urgency"],
        "financial": matches["financial"],
        "authority": matches["authority"],
    }


if __name__ == "__main__":
    examples = [
        "URGENT: Your bank payment must be completed immediately.",
        "Please contact IT support regarding your account.",
        "This is a normal project update for the team.",
    ]

    for example in examples:
        print()
        print("Text:", example)
        print(detect_manipulation_signals(example))
