from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Example training data (expand as needed)
training_examples = [
    # Arithmetic
    ("What is 2 plus 2?", "arithmetic"),
    ("Calculate 5 times 7", "arithmetic"),
    ("Add 10 and 15", "arithmetic"),
    ("Subtract 8 from 20", "arithmetic"),
    # Geometry
    ("Area of a circle with radius 5", "geometry"),
    ("What is the perimeter of a rectangle?", "geometry"),
    ("Volume of a sphere", "geometry"),
    # Conversion
    ("Convert 100 Celsius to Fahrenheit", "conversion"),
    ("What is 32 F in Celsius?", "conversion"),
    ("Change 50 degrees to Fahrenheit", "conversion"),
    # Assignment
    ("Let x = 5", "assignment"),
    ("Set y to 10", "assignment"),
    ("a = 7", "assignment"),
    # Equation
    ("Solve x + 2 = 5", "equation"),
    ("Find x if 2x = 10", "equation"),
    ("x^2 + 3x + 2 = 0", "equation"),
    # Fallback
    ("Tell me a joke", "other"),
    ("Who are you?", "other"),
]

X_train = [ex[0] for ex in training_examples]
y_train = [ex[1] for ex in training_examples]

# Train the model
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_train)
clf = LogisticRegression(max_iter=200)
clf.fit(X_vec, y_train)

def sklearn_predict_intent(user_input):
    """
    Predicts the intent of the user input using the trained sklearn model.
    Returns (intent_label, confidence_score)
    """
    X_test = vectorizer.transform([user_input])
    proba = clf.predict_proba(X_test)[0]
    idx = np.argmax(proba)
    label = clf.classes_[idx]
    confidence = proba[idx]
    return label, confidence

# Example usage:
if __name__ == "__main__":
    print("Intent Classifier (type 'exit' or 'quit' to stop)")
    while True:
        try:
            s = input("Enter query: ").strip()
            if not s:
                print("Please enter a non-empty query.")
                continue
            if s.lower() in ("exit", "quit"):
                print("Exiting.")
                break
            label, conf = sklearn_predict_intent(s)
            print(f"Intent: {label} (confidence: {conf:.2f})")
            if conf < 0.5:
                print("Warning: Low confidence. The intent may not be accurate.")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")
