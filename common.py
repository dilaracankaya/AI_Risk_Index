from warnings import filterwarnings
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tweepy
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import Word, TextBlob
from wordcloud import WordCloud
from dotenv import load_dotenv
import os
filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

load_dotenv()  # Load environment variables from .env file

api_key = os.getenv('api_key')
api_key_secret = os.getenv('api_key_secret')
access_token = os.getenv('access_token')
access_token_secret = os.getenv('access_token_secret')
bearer_token = os.getenv('bearer_token')



# Set up Tweepy client for API v2
client = tweepy.Client(bearer_token=bearer_token)



# FUNCTIONS
def fetch_tweets(query, count=100):
    tweet_list = []
    next_token = None
    remaining_tweets = count

    while remaining_tweets > 0:
        max_results = min(100, remaining_tweets)
        try:
            response = client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'text', 'author_id', 'public_metrics'],
                expansions=['author_id'],
                user_fields=['public_metrics'],
                next_token=next_token)

            if not hasattr(response, 'data'):
                break

            user_dict = {user.id: user for user in response.includes['users']} if 'users' in response.includes else {}

            for tweet in response.data:
                retweets = tweet.public_metrics.get('retweet_count', None)
                likes = tweet.public_metrics.get('like_count', None)
                user_info = user_dict.get(tweet.author_id)
                followers = user_info.public_metrics.get('followers_count', None) if user_info and hasattr(user_info,
                                                                                                           'public_metrics') else None

                tweet_info = {
                    "Date": tweet.created_at,
                    "User": tweet.author_id,
                    "Tweet": tweet.text,
                    "Retweets": retweets,
                    "Likes": likes,
                    "Followers": followers
                }
                tweet_list.append(tweet_info)

            next_token = response.meta.get('next_token', None)
            if not next_token:
                break

            remaining_tweets -= max_results

        except tweepy.TooManyRequests:
            print("Rate limit exceeded. Wait and try again.")
            break
        except tweepy.errors.Forbidden as e:
            print("Forbidden Error: ", e)
            break
        except Exception as e:
            print("An error occurred: ", e)
            break

    df = pd.DataFrame(tweet_list, columns=["Date", "User", "Tweet", "Retweets", "Likes", "Followers"])
    return df

def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity

def classify_tweet(tweet, safety_keywords, capability_keywords):
    has_capability = any(keyword in tweet for keyword in capability_keywords)
    has_safety = any(keyword in tweet for keyword in safety_keywords)

    if has_capability and has_safety:
        return "Both"
    elif has_capability:
        return "Capabilities"
    elif has_safety:
        return "Safety"
    else:
        return "Neither"

def process_df(analysis):
    if analysis == "psa":
        dataframe = tweets_df_psa
        safety_keywords = general_safety_alignment_terms + technical_safety_terms
    if analysis == "rsa":
        dataframe = tweets_df_rsa
        safety_keywords = general_safety_alignment_terms + regulatory_policy_terms

    # Normalizing Case Folding
    dataframe['Tweet'] = dataframe['Tweet'].str.lower()

    # Punctuations
    dataframe['Tweet'] = dataframe['Tweet'].str.replace('[^\w\s]', '', regex=True)

    # Numbers
    dataframe['Tweet'] = dataframe['Tweet'].str.replace('\d', '')

    # Stopwords
    # nltk.download('stopwords')
    sw = stopwords.words('english')
    dataframe['Tweet'] = dataframe['Tweet'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

    # Rarewords
    # TODO cancelled Rareword processing because it's a small dataset and I don't want to lose words
    # temp_df = pd.Series(' '.join(dataframe['Tweet']).split()).value_counts()
    # drops = temp_df[temp_df <= 1]
    # dataframe['Tweet'] = dataframe['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

    # Lemmatization
    # nltk.download('wordnet')
    dataframe['Tweet'] = dataframe['Tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    dataframe["polarity_score"] = dataframe["Tweet"].apply(lambda x: sia.polarity_scores(x)["compound"])

    dataframe["Classification"] = dataframe["Tweet"].apply(classify_tweet, args=(safety_keywords, capability_keywords))

    # Stance_-1to1 is the compound score of SIA, adjusted to be POS for pro-AI-Capabilities and NEG for pro-AI-Safety
    dataframe["Stance_-1to1"] = dataframe.apply(
        lambda row: -row["polarity_score"] if row["polarity_score"] > 0 and row["Classification"] == "Safety" else row[
            "polarity_score"] if row["polarity_score"] < 0 and row["Classification"] == "Capabilities" else row[
            "polarity_score"], axis=1)

    # Normalize Stance_-1to1 to between 0 to 100
    dataframe["Stance_0to100"] = ((dataframe["Stance_-1to1"] + 1) / 2 * 100).round(2)
    dataframe["Stance_weighted_reach"] = dataframe["Stance_0to100"] * np.log(dataframe['Followers'] + 1)
    # TODO add line to accommodate retweet count bc I will pull once more without RTs but with just the RT numbers
    return dataframe

def recalculate_reclassification(dataframe):
    dataframe["Stance_-1to1"] = dataframe.apply(
        lambda row: -row["polarity_score"] if row["polarity_score"] > 0 and row["Classification"] == "Safety" else row[
            "polarity_score"] if row["polarity_score"] < 0 and row["Classification"] == "Capabilities" else row[
            "polarity_score"], axis=1)

    # Normalize Stance_-1to1 to between 0 to 100
    dataframe["Stance_0to100"] = ((dataframe["Stance_-1to1"] + 1) / 2 * 100).round(2)
    dataframe["Stance_weighted_reach"] = dataframe["Stance_0to100"] * np.log(dataframe['Followers'] + 1)

    return dataframe

def change_classification(dataframe, search_string, new_classification):
    dataframe.loc[dataframe["Tweet"].str.contains(search_string, case=False), "Classification"] = new_classification

def save_to_csv(dataframe, filename):
    dataframe.to_csv(filename, index=False)



date_dmy = datetime.today().strftime('%d%m%y')


# Dictionaries of keywords related to AI capabilities and AI safety
capability_keywords = [
    'innovation', 'innovative', 'progress', 'progressive', 'technology', 'technological', 'capability', 'capable', 'efficiency', 'efficient', 'productivity', 'productive',
    'AI', 'artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'algorithm', 'model', 'training', 'trained',
    'GPT', 'transformer', 'BERT', 'T5', 'finetune', 'finetuning', 'finetuned', 'pretraining', 'pretrained', 'language model', 'supervised learning',
    'unsupervised learning', 'reinforcement learning', 'selfsupervised learning', 'generative', 'generation', 'autonomous', 'automation', 'selflearning',
    'scalability', 'scalable', 'generalization', 'generalize', 'optimization', 'optimized', 'intelligence', 'intelligent', 'AGI', 'artificial general intelligence', 'advanced AI',
    'breakthrough', 'disruptive', 'disruption', 'stateoftheart', 'cuttingedge', 'future', 'futuristic', 'evolution', 'evolved', 'enhancement', 'enhanced',
    'adaptability', 'adaptive', 'metalearning', 'fewshot learning', 'zeroshot learning', 'natural language processing', 'NLP', 'contextual understanding',
    'contextaware', 'semantic', 'syntactic', 'syntax', 'dialogue system', 'conversational AI', 'multimodal', 'multimodality', 'personalization', 'personalized',
    'automated reasoning', 'logical reasoning', 'cognitive', 'cognition', 'perception', 'predictive', 'prediction', 'datadriven', 'datacentric', 'big data', 'data analysis',
    'data analytic', 'data science', 'knowledge graph', 'knowledge representation', 'ontology', 'embedding', 'vector space', 'dimensionality',
    'dimensionality reduction', 'latent space', 'latent representation', 'feature engineering', 'feature extraction', 'attention mechanism', 'attention', 'crossattention', 'selfattention',
    'encoder', 'decoder', 'encoderdecoder', 'bidirectional', 'contextual embedding', 'word embedding', 'sentence embedding', 'tokenization', 'token', 'preprocessing',
    'postprocessing', 'inference', 'inferencing', 'validation', 'validation set', 'test set', 'benchmark', 'benchmarking', 'evaluation', 'metric', 'performance', 'accuracy', 'precision', 'recall',
    'F1 score', 'ROC curve', 'AUC', 'hyperparameter', 'tuning', 'grid search', 'random search', 'bayesian optimization', 'metaoptimization', 'transfer learning',
    'knowledge transfer', 'domain adaptation', 'domain generalization', 'general AI', 'artificial intelligence capability']

general_safety_alignment_terms = [
    'alignment', 'aligned', 'robust', 'robustness', 'bias', 'biased', 'fair', 'fairness', 'interpretable', 'interpretability', 'explainable', 'explainability', 'transparency', 'transparent', 'accountable', 'accountability', 'oversee', 'oversight', 'trustworthiness', 'trustworthy', 'adversarial',
    'mitigation', 'compliance', 'compliant', 'standard', 'audit', 'governance', 'best practice', 'ethical AI', 'AI principle', 'humanintheloop', 'safe', 'safety', 'alignment', 'align', 'aligned']

technical_safety_terms = [
    'model drift', 'dataset bias', 'overfitting', 'underfitting', 'training data', 'validation data', 'prompt engineering',
    'reinforcement learning', 'reward modeling', 'human feedback', 'differential privacy', 'federated learning', 'adversarial attack', 'defense mechanism', 'robustness check',
    'anomaly detection', 'safety protocol', 'kill switch', 'failsafe', 'control system', 'constraint satisfaction', 'value alignment', 'scalable oversight', 'corrigibility',
    'interpretability tool', 'posthoc analysis', 'causality', 'counterfactual reason', 'predictive accuracy', 'performance metric']

regulatory_policy_terms = [
    'AI policy', 'legislative framework', 'compliance standard', 'AI ethic board', 'regulatory compliance', 'risk assessment', 'safety standard', 'policy recommendation',
    'AI oversight', 'international cooperation', 'AI treaty']

# first keyword lists:
# capability_keywords = ['innovation', 'innovative', 'progress', 'progressive', 'technology', 'technological', 'capability', 'capable', 'efficiency', 'efficient', 'productivity', 'productive']
# safety_keywords = ['safe', 'safety', 'risk', 'ethic', 'ethical', 'regulate', 'regulation', 'secure', 'security', 'harm', 'control']

"""
def score_tweet(tweet, capability_keywords, safety_keywords):
    capability_score = sum(1 for word in capability_keywords if word in tweet.lower())
    safety_score = sum(1 for word in safety_keywords if word in tweet.lower())
    return capability_score, safety_score

def classify_sentiment(polarity): # TODO do I really need this? -- can I not use sia?
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

def classify_stance(capability_score, safety_score):
    if capability_score > safety_score:
        return 'AI Capabilities'
    elif safety_score > capability_score:
        return 'AI Safety'
    else:
        return 'Neutral'
"""
"""
tweets_df["Tweet"].iloc[5392]
df["polarity_score"].iloc[3078]
df["Stance_0to100"].iloc[3078]

condition = (((df["Classification"] == "Both") | (df["Classification"] == "Neither")) & ((df["Stance_0to100"] < 35) | (df["Stance_0to100"] > 65)))
indexes = df[condition].index
df_loc = tweets_df.loc[indexes, ["Tweet", "Retweets"]]
df_loc.shape

condition1 = ((df["Classification"] == "Neither") & (df["Stance_0to100"] < 35))
indexes1 = df[condition1].index
df_loc1 = tweets_df.loc[indexes1, ["Tweet", "Retweets"]]

df["Stance_-1to1"].iloc[2825]
df["Stance_0to100"].iloc[2825]
df["Classification"].iloc[3078]
df["Tweet"].iloc[3078]
"""
"""
#### buraya kadar çalıştırdım, error veriyor çünkü access yok.

# Prepare CSV file
with open('tweets.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Tweet", "Sentiment"])

    # Process each tweet
    for tweet in tweets.data:
        # Perform sentiment analysis
        analysis = TextBlob(tweet.text)
        sentiment = analysis.sentiment.polarity

        # Write to CSV
        writer.writerow([tweet.text, sentiment])

print("Tweets have been saved to tweets.csv")

### kısa blok bitiş
"""
"""
# Plot stance over time
tweets_df['Date'] = pd.to_datetime(tweets_df['Date'])
tweets_df.set_index('Date', inplace=True)
tweets_df.resample('D').size().unstack().plot(kind='bar', stacked=True)
plt.title('Stance Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.show()

# Plot sentiment over time
tweets_df['Date'] = pd.to_datetime(tweets_df['Date'])
tweets_df.set_index('Date', inplace=True)
tweets_df.resample('D').mean()['Sentiment'].plot()
plt.title('Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.show()
"""
"""
############### burası da ilk kodum

# Define your queries
queries = [
    "AI policy",
    "AI ethics",
    "AI regulation",
    "AI safety",
    "AI governance framework",
    "#AIgovernance",
    "#AIethics",
    "#AIpolicy"
    "AI law"
]

# Fetch tweets for each query and save to CSV
for query in queries:
    tweets_df = fetch_tweets(query, count=100)
    if not tweets_df.empty:
        # Clean the query string for filename
        query_filename = query.replace(" ", "_").replace("#", "")
        save_to_csv(tweets_df, f"tweets_{query_filename}.csv")
    else:
        print(f"No tweets found for query: {query}")
"""
"""
# Fetch tweets
safe_query = "AI governance" # TODO need to change this
safe_tweets_df = fetch_tweets(safe_query, count=100)
safe_tweets_df.to_csv("safe_tweets.csv", index=False)

cap_query = "AI capabilities" # TODO need to change this
cap_tweets_df = fetch_tweets(cap_query, count=100)
safe_tweets_df.to_csv("cap_tweets.csv", index=False)

# Analyze sentiment
tweets_df['Sentiment'] = tweets_df['Tweet'].apply(analyze_sentiment)
tweets_df['Sentiment_Class'] = tweets_df['Sentiment'].apply(classify_sentiment)

# Score tweets
tweets_df[['Capability_Score', 'Safety_Score']] = tweets_df['Tweet'].apply(lambda tweet: pd.Series(score_tweet(tweet, capability_keywords, safety_keywords)))
tweets_df['Stance'] = tweets_df.apply(lambda row: classify_stance(row['Capability_Score'], row['Safety_Score']), axis=1)
"""