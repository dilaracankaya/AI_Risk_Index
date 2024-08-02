import os
import numpy as np
import pandas as pd
import tweepy
import csv
import calendar
from datetime import datetime, timedelta
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import Word, TextBlob
from warnings import filterwarnings
filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# ! "If a search request does not specify a start_time, end_time, or since_id request parameter, the end_time will default to "now" (actually 30 seconds before the time of query) and the start_time will default to seven days ago."

load_dotenv("credentials.env")

api_key = os.getenv('api_key')
api_key_secret = os.getenv('api_key_secret')
access_token = os.getenv('access_token')
access_token_secret = os.getenv('access_token_secret')
bearer_token = os.getenv('bearer_token')

client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=api_key,
    consumer_secret=api_key_secret,
    access_token=access_token,
    access_token_secret=access_token_secret)


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
                followers = user_info.public_metrics.get('followers_count', None) if user_info and hasattr(user_info, 'public_metrics') else None

                tweet_info = {
                    "Date": tweet.created_at,
                    "User": tweet.author_id,
                    "Tweet": tweet.text,
                    "Retweets": retweets,
                    "Likes": likes,
                    "Followers": followers}
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def process_df(analysis_type, dataframe):
    if analysis_type == "psa":
        safety_keywords = general_safety_alignment_terms + technical_safety_terms
    elif analysis_type == "rsa":
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

    # Rarewords -- cancelled Rareword processing because it's a small dataset and I don't want to lose words
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

    influence_factor = sigmoid(np.log(dataframe['Followers'] + 1) + np.log(dataframe['Retweets'] + 1))
    dataframe["Stance_weighted_reach_-1to1"] = (dataframe["Stance_-1to1"] * influence_factor)
    dataframe["Stance_weighted_0to100"] = ((dataframe["Stance_weighted_reach_-1to1"] + 1) / 2 * 100).round(2)
    return dataframe


def recalculate_reclassification(dataframe):
    dataframe["Stance_-1to1"] = dataframe.apply(
        lambda row: -row["polarity_score"] if row["polarity_score"] > 0 and row["Classification"] == "Safety" else row[
            "polarity_score"] if row["polarity_score"] < 0 and row["Classification"] == "Capabilities" else row[
            "polarity_score"], axis=1)

    influence_factor = sigmoid(np.log(dataframe['Followers'] + 1) + np.log(dataframe['Retweets'] + 1))
    dataframe["Stance_weighted_reach_0to1"] = (dataframe["Stance_-1to1"] * influence_factor).round(2)
    dataframe["Stance_0to100"] = (dataframe["Stance_weighted_reach_0to1"] * 100).round(2)
    return dataframe


def change_classification(dataframe, search_string, new_classification):
    dataframe.loc[dataframe["Tweet"].str.contains(search_string, case=False), "Classification"] = new_classification


def save_to_csv(dataframe, filename):
    dataframe.to_csv(filename, index=False)


def update_csv_and_list(new_value, csv_name):
    csv_name = f"historical_data/{csv_name}.csv"

    # Update historical csv with newly calculated final_psa
    fields = [end_date, new_value]
    with open(rf"{csv_name}", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    updated_hist_list = []
    # Read all data from the CSV file
    with open(csv_name, "r") as f:
        reader = csv.reader(f)
        headers = next(reader, None)  # Read and skip the header row
        for row in reader:
            if len(row) > 1:  # Ensure the row has at least two columns
                try:
                    updated_hist_list.append(float(row[1]))  # Convert the second column to float and append
                except ValueError:
                    pass  # Skip rows where conversion fails

    return updated_hist_list


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

# DATE TO SAVE DATASETS
# date_dmy = datetime.today().strftime('%d%m%y')
date_dmy = "010824"

today = datetime.today()
if today.weekday() == 0:  # Monday is 0
    end_date = today
else:
    end_date = today - timedelta(days=today.weekday())


# FETCHING TWEETS
# Calculate the number of tweets to pull weekly
monthly_tweet_topull = 10000

# How many tweets should be pulled now? Calculate the number of Mondays in the current month
year = datetime.today().year
month = datetime.today().month
num_mondays = sum(1 for day in calendar.monthcalendar(year, month) if day[calendar.MONDAY] != 0)

weekly_tweet_topull = monthly_tweet_topull / num_mondays
weekly_psa_topull = weekly_rsa_topull = weekly_tweet_topull/2


# REGULATORY SENTIMENT ANALYSIS
query_rsa = "AI (regulation OR regulations OR regulatory OR policy OR policies OR framework OR frameworks OR government OR governance OR legislation OR laws OR compliance OR oversight OR standards OR ethics OR ethical OR guidelines OR safety OR audit OR accountability OR transparency OR risk OR management OR assessment OR cooperation OR treaty OR treaties) -is:retweet lang:en -decentralized -DAO -crypto -cryptocurrency -cryptocurrencies -#crypto -#cryptocurrency -L2 -#L2 -#Layer2"
file_path = f"fetched_tweets/tweets_rsa_{date_dmy}.csv"

if not os.path.exists(file_path):
    tweets_df_rsa = fetch_tweets(query_rsa, count=weekly_rsa_topull)
    save_to_csv(tweets_df_rsa, file_path)
else:
    tweets_df_rsa = pd.read_csv(file_path)
    tweets_df_rsa.head()

process_df("rsa", tweets_df_rsa).head()
tweets_df_rsa["Stance_weighted_0to100"] = tweets_df_rsa["Stance_weighted_0to100"].round(2)
tweets_df_rsa["Stance_weighted_0to100"].describe().T

# Manual tweaks specific to this dataset
"""
#df.loc[df["Tweet"].str.contains("firstever global ai summit last year")]
change_classification(tweets_df_rsa, "exploited consumer bluwhale leverage", "Safety")
recalculate_reclassification(tweets_df_rsa)
"""
# tweets_df_rsa.groupby("Classification")["Stance_0to100"].agg(["count", "mean"])

# Final score of RSA
final_rsa = round(tweets_df_rsa["Stance_weighted_0to100"].mean(), 2)
hist_rsa = update_csv_and_list(final_rsa, "hist_rsa")


# PUBLIC SENTIMENT ANALYSIS
query_psa = "AI (governance OR policy OR regulation OR capabilities) OR (#AIgovernance OR #AIpolicy) -is:retweet lang:en -decentralized -DAO -blockchain -#blockchain -blockchainassn -crypto -cryptocurrency -cryptocurrencies -#crypto -#cryptocurrency -L2 -#L2 -#Layer2"
file_path = f"fetched_tweets/tweets_psa_{date_dmy}.csv"

if not os.path.exists(file_path):
    tweets_df_psa = fetch_tweets(query_psa, count=weekly_psa_topull)
    save_to_csv(tweets_df_psa, file_path)
else:
    tweets_df_psa = pd.read_csv(file_path)
    tweets_df_psa.head()

process_df("psa", tweets_df_psa).head()
tweets_df_psa["Stance_weighted_0to100"].describe().T

# Manual tweaks specific to this dataset
"""
#df.loc[df["Tweet"].str.contains("firstever global ai summit last year")]
change_classification(tweets_df_psa, "cess_storage excited join", "Capabilities")
change_classification(tweets_df_psa, "firstever global ai summit last year", "Safety")
recalculate_reclassification(tweets_df_psa)
"""
# tweets_df_psa.groupby("Classification")["Stance_0to100"].agg(["count", "mean"])

# Final score of PSA
final_psa = round(tweets_df_psa["Stance_weighted_0to100"].mean(), 2)
hist_psa = update_csv_and_list(final_psa, "hist_psa")



# INVESTMENTS
# TODO should these also use update_csv_and_list? Because unless saved into a csv, the data and date calculated separately may fall out of sync.
hist_invcap = [95.39, 95.35, 97.02, 97.71, 97.67, 97.03, 95.93, 96.04, 96.03, 94.96, 94.84, 94.01, 95.29, 94.44]
hist_invsaf = [17.47, 18.69, 19.58, 24.13, 19.44, 14.70, 15.79, 11.08, 11.55, 11.57, 21.16, 30.41, 30.76, 26.54]
final_invcap = hist_invcap[-1]
final_invsaf = hist_invsaf[-1]

print(len(hist_rsa), len(hist_psa), len(hist_invcap), len(hist_invsaf))
# FINAL TOTAL CALCULATION
hist_airi = [round(0.25 * (hist_rsa[i] + hist_psa[i] + hist_invcap[i] + hist_invsaf[i]), 2) for i in range(len(hist_rsa))]
airi_score = round(0.25*(final_psa + final_rsa + final_invcap + final_invsaf), ndigits=2)
