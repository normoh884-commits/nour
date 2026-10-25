import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def load_dummy_data():
    '''Generates sample text data to ensure the script runs immediately.'''
    data = {
        'text': [
            "Congratulations! You won a $1000 gift card. Click here to claim.",
            "Hey, are we still meeting for lunch tomorrow at 2 PM?",
            "URGENT: Your bank account has been suspended. Verify your identity now.",
            "Please find attached the report for Q3 performance.",
            "Win free tickets to the upcoming concert! Reply WIN to this email.",
            "Can you review the pull request on GitHub when you have a moment?",
            "You have been selected for a free vacation. Call now!",
            "Don't forget to submit your assignment by Friday.",
            "Limited time offer! Get 50% off on all items.",
            "Let's schedule a call to discuss the new n8n workflow integration."
        ],
        'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("📊 Loading Data...")
    df = load_dummy_data()

    print("🧹 Preprocessing Data (NLP)...")
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

    # Convert text to numerical vectors using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("🤖 Training Model (Naive Bayes)...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    print("📈 Evaluating Model...")
    predictions = model.predict(X_test_vec)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\n💡 Business Note: High Precision is prioritized to avoid sending important emails to the spam folder!")
