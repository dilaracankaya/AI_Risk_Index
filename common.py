from warnings import filterwarnings
import numpy as np
import pandas as pd
import tweepy
import os
import base64
import tempfile
import calendar
from datetime import datetime, timedelta
from dotenv import load_dotenv
import shutil
import subprocess
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import Word, TextBlob
filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


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
