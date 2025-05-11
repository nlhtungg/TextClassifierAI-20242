import random

def predict(text: str):
    # Example news categories
    categories = ["politics", "sports", "entertainment", "technology", "business"]

    # Select a random category and generate a random confidence score between 0.5 and 1.0
    prediction = random.choice(categories)

    return prediction

if __name__ == '__main__':
    sample_text = "Sample content of a news article to classify."
    result = predict(sample_text)
    print("News Category Prediction:", result)