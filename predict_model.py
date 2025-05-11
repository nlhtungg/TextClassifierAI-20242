import random
import time

def predict(text: str):
    # Simulate processing delay (1-3 seconds)
    delay = random.uniform(1, 3)
    time.sleep(delay)
    
    # Example news categories
    categories = ["politics", "sports", "entertainment", "technology", "business"]

    # Select a random category
    prediction = random.choice(categories)
    
    return {
        "category": prediction,
        "processing_time": round(delay, 2)
    }

if __name__ == '__main__':
    sample_text = "Sample content of a news article to classify."
    result = predict(sample_text)
    print("News Category Prediction:", result)