# NaiveBayes-ProbStat-W3

# Kelompok 10
1. 5025241058 - Addien Zafriyan Al Akhsan
2. 5025241104 - Raden Kurniawan Agung Fitrianto
3. 5025241107 - Muhammad Zahran Rizki Primanda
4. 5025241162 - Felix Aldorino

# Bayes' theorem Introduction

Bayes' theorem  is used to determine the conditional probability of an event when another event has already occurred.  

The Formula is:  

P(A|B)=P(A) P(B|A)P(B)  

Which tells us: how often A happens given that B happens, written P(A|B)  
When we know:   
how often B happens given that A happens, written P(B|A)  
how likely A is on its own, written P(A)  
how likely B is on its own, written P(B)  

#  Bayes' theorem on Naive Bayes

Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable.   

Practical Example: Spam Detector  

Bayes Theorem: Focuses on finding the probability of a category (like "Spam") given a feature (like the word "Buy").
Naive Assumption: It assumes that all features are independent of each other. In reality, words like "Buy" and "Cheap" often appear together, but assuming they don't simplifies the math significantly.  

The Initial Data:  
Total Emails: 100  
Spam: 25 emails  
Not Spam (Ham): 75 emails  

The algorithm begins by calculating how individual keywords correlate with a category. Using the spam detector example:  
Word "Buy": Found in 20/25 spam emails and 5/75 legitimate ones. If an email has this word, it has an 80% spam probability.  
Word "Cheap": Found in 15/25 spam emails and 10/75 legitimate ones. This word alone gives a 60% spam probability.  

When we combine words (e.g., an email contains both "Buy" AND "Cheap"), we encounter a logical hurdle. In a small dataset, it is possible that the exact combination of   those two words has never appeared in a "Not Spam" email. If we rely strictly on exact matches, the math would suggest a 100% certainty of spam. This is inaccurate because   it assumes that just because a combination hasn't been seen yet, it can never exist in a legitimate email.  

To resolve this, Naive Bayes stops looking for exact combinations and instead estimates them using the Independence Assumption. It treats each word as if it has no   relationship with the others:  

Instead of searching for "Buy + Cheap" as a pair, the algorithm multiplies the individual probability of "Buy" by the probability of "Cheap."  
By multiplying these individual stats against the total number of emails, we can estimate that an email containing both words has a 94.7% probability of being spam. 

In conclusion, this method is called "naive" because it simplistically assumes that every feature or word is completely independent and does not influence any other, even   though keywords are often related in real life. Despite this "naive" and theoretically flawed assumption, the algorithm remains highly popular because it is incredibly   fast, computationally efficient, and remarkably accurate for large-scale text classification tasks like spam filtering and sentiment analysis.  

# DATASET

The dataset used for this implementation is sourced from [Kaggle](https://www.kaggle.com/datasets/vstepanenko/disaster-tweets/data) which presents a collection of tweets from January 14th 2020 associated with disaster keywords like “crash”, “quarantine”, and “bush fires” as well as the location and keyword itself.  

Though many of the tweets seems to be discussing about a real disaster or tragedy unfolding, some of them might have been a product of fearmongering or hyperbole and as a result might not depict any real incident happening in the world and instead be of people over exxagerating certain aspects that happen in life to the internet to get clicks and views.  

That is why, using the Bayes Theorem to calculate the probability of events from the data being actual real disasters or not.

The structure of the dataset itself is as follows:

| id | keyword | location | text | target |
| --- | --- | --- | --- | --- |
| A unique identifier for each tweet | A particular keyword from the tweet | The location the tweet was sent from (may be blank) | The text of the tweet | Denotes whether a tweet is about a real disaster (1) or not (0) |

For our program, we only use the text column (with some cleaning) as features and target column as label.

# Code Explanation

#### Importing Libraries
```
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
```
#### Load dataset into a table variable called df
```
df = pd.read_csv('tweets.csv')
```

#### Function to strip out URLs that don’t help the model understand sentiment and converts everything to lowercase
```
def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    return text.lower()
```

#### Handle NULL value in the text column
```
df['text'] = df['text'].fillna('')
```

#### Clean the text by inputting the text column into the clean_text function and creating a new column called clean_text to store the cleaned text
```
df['clean_text'] = df['text'].apply(clean_text)
```

#### X is the cleaned text and y is the target variable (0 or 1) indicating whether the tweet is about a real disaster or not
```
X = df['clean_text']
y = df['target']
```

#### Split the data into training and testing sets (80% train, 20% test)
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Counts how many times every unique word appears in every tweet and removes common English stop words
```
vectorizer = CountVectorizer(stop_words='english')
```

#### Transforms the text data into a matrix of token counts for both training and testing sets
```
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```

#### Define the Multinomial Naive Bayes classifier
```
nb_classifier = MultinomialNB()
```

#### Fit the model to the training data
```
nb_classifier.fit(X_train_vec, y_train)
```

#### Predict the target variable for the test set
```
y_pred = nb_classifier.predict(X_test_vec)
```

#### Evaluate the model's performance by calculating the accuracy and displaying the confusion matrix
```
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

### Predicting New Data
#### Define new tweets to classify
```
new_tweets = [
    "Emergency services are on the scene of a massive multi-car pileup on I-95.",
    "That concert last night was total chaos, my heart is literally on fire!",
    "Fire fighters are battling a blaze that has engulfed a residential building downtown.",
    "Earthquake of magnitude 6.5 strikes near the coast, causing widespread damage and panic.",
    "Flooding in my kitchen right now, send help!",
    "fire damage panic public explosion disaster emergency nuclear bomb blast tsunami earthquake hurricane tornado wildfire",
]
```

#### Transform the text into the same format as the training data
```
new_samples_vec = vectorizer.transform(new_tweets)
```

#### Predict the target (0 or 1)
```
predictions = nb_classifier.predict(new_samples_vec)
```

#### Get the confidence scores
```
probabilities = nb_classifier.predict_proba(new_samples_vec)
```

#### Display results
```
for tweet, pred, prob in zip(new_tweets, predictions, probabilities):
    label = "Real Disaster" if pred == 1 else "Not a Disaster"
    confidence = prob[pred] * 100
    print(f"Tweet: {tweet}")
    print(f"Prediction: {label} ({confidence:.2f}% confidence)\n")
```

# Naive Bayes Execution Evalution

Based on the execution of Naive Bayes algorithme on the dataset, we have arrive with 89.40% accurancy regarding the classification of the probability of events from the data being actual real disasters or not  
Example:   
Tweet: Emergency services are on the scene of a massive multi-car pileup on I-95.   
Prediction: Real Disaster (99.70% confidence)  
Tweet: That concert last night was total chaos, my heart is literally on fire!  
Prediction: Not a Disaster (99.91% confidence)  
Tweet: Fire fighters are battling a blaze that has engulfed a residential building downtown.  
Prediction: Real Disaster (80.69% confidence)  
Tweet: Earthquake of magnitude 6.5 strikes near the coast, causing widespread damage and panic.  
Prediction: Real Disaster (99.38% confidence)  
Tweet: Flooding in my kitchen right now, send help!  
Prediction: Not a Disaster (98.14% confidence)  

Based on the results, we conclude that the algorithm achieves a very high level of correct classification for real disasters. Although some classifications have lower confidence levels, they still produce the correct results. The algorithm is also able to effectively distinguish between major real disasters and event that are accidents but not a disaster with remarkably high confidence.
