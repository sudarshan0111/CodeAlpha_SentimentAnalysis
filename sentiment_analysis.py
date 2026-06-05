import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("IMDB-Dataset.csv")

# Use first 1000 reviews for faster execution
df = df.head(1000)

print("Dataset Shape:", df.shape)
print(df.head())
print("\nColumns:", df.columns)

# Sentiment Analysis Function
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply Sentiment Analysis
df["Predicted_Sentiment"] = df["review"].apply(get_sentiment)

# Display Sample Results
print("\nSample Results:")
print(df[["review", "Predicted_Sentiment"]].head())

# Count Sentiments
sentiment_count = df["Predicted_Sentiment"].value_counts()

print("\nSentiment Counts:")
print(sentiment_count)

# Pie Chart
plt.figure(figsize=(6,6))

plt.pie(
    sentiment_count,
    labels=sentiment_count.index,
    autopct="%1.1f%%"
)

plt.title("Sentiment Distribution")

plt.savefig("sentiment_pie_chart.png")

plt.show()

# Bar Chart
plt.figure(figsize=(6,5))

sns.countplot(
    x="Predicted_Sentiment",
    data=df
)

plt.title("Sentiment Analysis Results")

plt.savefig("sentiment_bar_chart.png")

plt.show()

# Save Results
df.to_csv(
    "sentiment_results.csv",
    index=False
)

print("\nSentiment Analysis Completed Successfully!")
print("Files Generated:")
print("1. sentiment_pie_chart.png")
print("2. sentiment_bar_chart.png")
print("3. sentiment_results.csv")