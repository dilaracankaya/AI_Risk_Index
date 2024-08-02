from common import *
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


# Calculate the number of tweets to pull weekly
monthly_tweet_topull = 10000

# How many tweets should be pulled now? Calculate the number of Mondays in the current month
year = datetime.today().year
month = datetime.today().month
num_mondays = sum(1 for day in calendar.monthcalendar(year, month) if day[calendar.MONDAY] != 0)

weekly_tweet_topull = monthly_tweet_topull / num_mondays
weekly_psa_topull = weekly_rsa_topull = weekly_tweet_topull/2

# date_dmy = datetime.today().strftime('%d%m%y')
date_dmy = "010824"


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
hist_rsa = [40.12, 42.15, 43.67, 44.88, 46.20, 47.35, 48.58, 49.12, 49.87, 50.24, 50.67, 51.22, 50.89]
final_rsa = round(tweets_df_rsa["Stance_weighted_0to100"].mean(), 2)
hist_rsa.append(final_rsa)


## PUBLIC SENTIMENT ANALYSIS
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
hist_psa = [62.34, 63.56, 65.12, 66.78, 67.89, 68.45, 68.90, 69.12, 69.45, 69.78, 70.12, 70.50, 70.77]
final_psa = round(tweets_df_psa["Stance_weighted_0to100"].mean(), 2)
hist_psa.append(final_psa)


# INVESTMENTS
hist_invcap = [95.39, 95.35, 97.02, 97.71, 97.67, 97.03, 95.93, 96.04, 96.03, 94.96, 94.84, 94.01, 95.29, 94.44]
hist_invsaf = [17.47, 18.69, 19.58, 24.13, 19.44, 14.70, 15.79, 11.08, 11.55, 11.57, 21.16, 30.41, 30.76, 26.54]

final_invcap = hist_invcap[-1]
final_invsaf = hist_invsaf[-1]


# FINAL TOTAL CALCULATION
hist_airi = [round(0.25 * (hist_rsa[i] + hist_psa[i] + hist_invcap[i] + hist_invsaf[i]), 2) for i in range(len(hist_rsa))]
airi_score = round(0.25*(final_psa + final_rsa + final_invcap + final_invsaf), ndigits=2)
