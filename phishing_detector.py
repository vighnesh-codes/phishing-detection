import re
import ipaddress
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASET_PATH = "phishing_emails.csv"

SUSPICIOUS_KEYWORDS = [
    "urgent", "verify your account", "password", "click here",
    "account suspended", "account blocked", "confirm your identity",
    "free gift", "claim your prize", "bank details",
    "login immediately", "limited time", "wire transfer",
    "payment required", "security alert",
]

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "ow.ly"
}

URL_PATTERN = re.compile(r"""(?i)\b(?:https?://|www\.)[^\s<>"']+""")

FEATURE_NAMES = [
    "URL Count",
    "Suspicious Keywords",
    "IP Address URLs",
    "Shortened URLs",
    "HTTPS URLs",
    "Exclamation Marks",
    "Uppercase Characters",
    "Text Length",
    "Word Count",
]


def extract_urls(text):
    """Extract URLs from email text."""
    return URL_PATTERN.findall(str(text))


def is_ip_address_url(url):
    """Return 1 if the URL hostname is an IP address; otherwise 0."""
    try:
        candidate = url if url.startswith(("http://", "https://")) else "http://" + url
        hostname = urlparse(candidate).hostname
        if not hostname:
            return 0
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def extract_features(email):
    """Extract numerical features from email text."""
    email = str(email)
    email_lower = email.lower()
    urls = extract_urls(email)

    url_count = len(urls)
    suspicious_keyword_count = sum(
        email_lower.count(keyword) for keyword in SUSPICIOUS_KEYWORDS
    )
    ip_url_count = sum(is_ip_address_url(url) for url in urls)

    shortened_url_count = 0
    for url in urls:
        candidate = url if url.startswith(("http://", "https://")) else "http://" + url
        hostname = (urlparse(candidate).hostname or "").lower()
        if hostname in URL_SHORTENERS:
            shortened_url_count += 1

    https_url_count = sum(1 for url in urls if url.lower().startswith("https://"))
    exclamation_count = email.count("!")
    uppercase_count = sum(1 for character in email if character.isupper())
    text_length = len(email)
    word_count = len(email.split())

    return [
        url_count,
        suspicious_keyword_count,
        ip_url_count,
        shortened_url_count,
        https_url_count,
        exclamation_count,
        uppercase_count,
        text_length,
        word_count,
    ]


def normalize_labels(series):
    """Normalize common dataset label names to 1=phishing and 0=safe."""
    label_mapping = {
        "phishing": 1, "phish": 1, "1": 1,
        "safe": 0, "legitimate": 0, "ham": 0, "0": 0,
    }
    return series.astype(str).str.strip().str.lower().map(label_mapping)


def main():
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        raise SystemExit(
            f"Could not find {DATASET_PATH}. Run this script from the project folder."
        )

    required_columns = {"email_text", "label"}
    if not required_columns.issubset(df.columns):
        raise SystemExit(
            "Dataset must contain these columns: email_text,label"
        )

    df = df.dropna(subset=["email_text", "label"]).copy()
    df["email_text"] = df["email_text"].astype(str).str.strip()
    df["label"] = normalize_labels(df["label"])
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["email_text"] != ""]
    df = df.drop_duplicates(subset=["email_text"])

    if df["label"].nunique() != 2:
        raise SystemExit("Dataset must contain both phishing and safe emails.")
    if df["label"].value_counts().min() < 2:
        raise SystemExit("Each class needs at least two examples.")

    print("\nDataset loaded successfully!")
    print(f"Total emails: {len(df)}")
    print("\nClass distribution (0=Safe, 1=Phishing):")
    print(df["label"].value_counts().sort_index())

    X_text = df["email_text"].to_numpy()
    y = df["label"].to_numpy()
    X_numeric = np.array([extract_features(email) for email in X_text], dtype=float)

    X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = (
        train_test_split(
            X_text,
            X_numeric,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    X_train = hstack([X_train_tfidf, csr_matrix(X_train_scaled)], format="csr")
    X_test = hstack([X_test_tfidf, csr_matrix(X_test_scaled)], format="csr")

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n" + "=" * 50)
    print("PHISHING EMAIL DETECTION RESULTS")
    print("=" * 50)
    print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Safe", "Phishing"],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Safe", "Phishing"],
        yticklabels=["Safe", "Phishing"],
    )
    plt.title("Phishing Email Detection - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()

    def predict_email(email):
        text_features = vectorizer.transform([email])
        numeric_features = np.array([extract_features(email)], dtype=float)
        scaled_features = scaler.transform(numeric_features)
        combined_features = hstack(
            [text_features, csr_matrix(scaled_features)], format="csr"
        )

        prediction = int(model.predict(combined_features)[0])
        probabilities = model.predict_proba(combined_features)[0]
        phishing_index = list(model.classes_).index(1)
        phishing_probability = probabilities[phishing_index] * 100
        label = "Phishing" if prediction == 1 else "Safe"

        print("\n" + "-" * 40)
        print("EMAIL ANALYSIS")
        print("-" * 40)
        print("Email:", email)
        print("Prediction:", label)
        print(f"Estimated phishing probability: {phishing_probability:.2f}%")
        print("\nExtracted Features:")
        for name, value in zip(FEATURE_NAMES, extract_features(email)):
            print(f"{name}: {value}")
        print("-" * 40)

    sample_email = (
        "URGENT! Your account is suspended. "
        "Click here to verify your password immediately!"
    )
    predict_email(sample_email)

    while True:
        choice = input("\nTest another email? Enter Y to test or N to exit: ").strip().lower()
        if choice != "y":
            print("Thank you for using the Phishing Email Detector!")
            break
        email_input = input("\nEnter the email text: ").strip()
        if email_input:
            predict_email(email_input)
        else:
            print("Please enter some email text.")


if __name__ == "__main__":
    main()
