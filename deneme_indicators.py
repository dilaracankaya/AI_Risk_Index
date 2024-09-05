import twint
import pandas as pd
from collections import Counter







# FUNCTIONS

# ! "If a search request does not specify a start_time, end_time, or since_id request parameter, the end_time will default to "now" (actually 30 seconds before the time of query) and the start_time will default to seven days ago."
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

    # Make a copy of the raw tweets
    dataframe['Tweet_original'] = dataframe['Tweet']

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
#date_dmy = datetime.today().strftime('%y%m%d')
date_dmy = "240830"

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
# weekly_psa_topull = weekly_rsa_topull = weekly_tweet_topull/2
weekly_psa_topull = weekly_rsa_topull = 1000


# REGULATORY SENTIMENT ANALYSIS
query_rsa = "-is:retweet AI (regulation OR regulations OR regulatory OR policy OR policies OR framework OR frameworks OR government OR governance OR legislation OR law OR laws OR compliance OR oversight OR standards OR ethics OR ethical OR guidelines OR safety OR audit OR accountability OR transparency OR risk OR management OR assessment OR cooperation OR treaty OR treaties) lang:en -decentralized -DAO -crypto -cryptocurrency -cryptocurrencies -#crypto -#cryptocurrency -L2 -#L2 -#Layer2"
file_path = f"fetched_tweets/tweets_rsa_{date_dmy}.csv"

if os.path.exists(file_path):
    tweets_df_rsa = pd.read_csv(file_path)
    tweets_df_rsa.head()
# else:
#     tweets_df_rsa = fetch_tweets(query_rsa, count=weekly_rsa_topull)
#     save_to_csv(tweets_df_rsa, file_path)


process_df("rsa", tweets_df_rsa).head()
# save_to_csv(tweets_df_rsa, file_path)
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
# hist_rsa = update_csv_and_list(final_rsa, "hist_rsa")


# PUBLIC SENTIMENT ANALYSIS
query_psa = " -is:retweet AI (governance OR policy OR regulation OR capabilities) OR (#AIgovernance OR #AIpolicy) lang:en -decentralized -DAO -blockchain -#blockchain -blockchainassn -crypto -cryptocurrency -cryptocurrencies -#crypto -#cryptocurrency -L2 -#L2 -#Layer2"
file_path = f"fetched_tweets/tweets_psa_{date_dmy}.csv"

if os.path.exists(file_path):
    tweets_df_psa = pd.read_csv(file_path)
    tweets_df_psa.head()
# else:
#     tweets_df_psa = fetch_tweets(query_psa, count=weekly_psa_topull)
#     save_to_csv(tweets_df_psa, file_path)

process_df("psa", tweets_df_psa).head()
tweets_df_psa["Stance_weighted_0to100"] = tweets_df_psa["Stance_weighted_0to100"].round(2)
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


# INVESTMENTS
final_invcap = 91.99
final_invsaf = 33.23


# FINAL TOTAL CALCULATION
# hist_airi = [round(0.25 * (hist_rsa[i] + hist_psa[i] + hist_invcap[i] + hist_invsaf[i]), 2) for i in range(len(hist_rsa))]
airi_score = round(0.25*(final_psa + final_rsa + final_invcap + final_invsaf), ndigits=2)


# RECORD KEEPING
score_records = pd.read_csv('historical_data/all_scores.csv')

# Calculate the date of the Monday of the current week
current_date = datetime.now()
start_of_week = current_date - timedelta(days=current_date.weekday())

# Create a new row
new_row = {'Date': start_of_week.strftime('%d/%m/%Y'),
           'invcap_Indicator_Score': final_invcap,
           'invcsaf_Indicator_Score': final_invsaf,
           'rsa_Indicator_Score': final_rsa,
           'psa_Indicator_Score': final_psa,
           'AIRI_Aggregate': airi_score}

score_records = score_records._append(new_row, ignore_index=True)

output_file_path = f'historical_data/all_scores_{date_dmy}.csv'
score_records.to_csv(output_file_path, index=False)

# Read the updated CSV file
df_updated = pd.read_csv(output_file_path)

# Create lists from the respective columns
hist_invcap = df_updated['invcap_Indicator_Score'].tolist()
hist_invsaf = df_updated['invcsaf_Indicator_Score'].tolist()
hist_rsa = df_updated['rsa_Indicator_Score'].tolist()
hist_psa = df_updated['psa_Indicator_Score'].tolist()
hist_airi = df_updated['AIRI_Aggregate'].tolist()
hist_date_records = df_updated['Date'].tolist()

# Convert format of hist_date_records from 26/07/2024 to 26 Aug
hist_date_records_formatted = []
for date_str in hist_date_records:
    date_obj = datetime.strptime(date_str, '%d/%m/%Y')  # Convert to datetime object
    hist_date_records_formatted.append(date_obj.strftime('%b %d'))  # Format to 'DD Mon'


"""
# hist_psa scores deleted from the beginning to make lengths of rsa/psa lists equal to lengths of inv lists (14): 62.34
# hist_psa of week of aug 19: 61.48
hist_rsa = [42.15, 43.67, 44.88, 46.20, 47.35, 48.58, 49.12, 49.87, 50.24, 50.67, 51.22, 50.89, 59.12, 59.35, 62.44, 57.72]
hist_psa = [63.56, 65.12, 66.78, 67.89, 68.45, 68.90, 69.12, 69.45, 69.78, 70.12, 70.50, 70.77, 59.27, 60.00, 60.09, 60.26]
len(hist_rsa)
final_rsa = hist_rsa[-1]
final_psa = hist_psa[-1]

# TODO should these also use update_csv_and_list? Because unless saved into a csv, the data and date calculated separately may fall out of sync.
hist_invcap = [97.02, 97.71, 97.67, 97.03, 95.93, 96.04, 96.03, 94.96, 94.84, 94.01, 95.29, 95.19, 94.80, 94.18, 94.24]
hist_invsaf = [19.58, 24.13, 19.44, 14.70, 15.79, 11.08, 11.55, 11.57, 21.16, 30.41, 30.76, 19.82, 21.57, 22.72, 32.46]
len(hist_invcap)
final_invcap = hist_invcap[-1]
final_invsaf = hist_invsaf[-1]
"""