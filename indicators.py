from common import *
# ! "If a search request does not specify a start_time, end_time, or since_id request parameter, the end_time will default to "now" (actually 30 seconds before the time of query) and the start_time will default to seven days ago."


## REGULATORY SENTIMENT ANALYSIS
query_rsa = "AI (regulation OR regulations OR regulatory OR policy OR policies OR framework OR frameworks OR government OR governance OR legislation OR laws OR compliance OR oversight OR standards OR ethics OR ethical OR guidelines OR safety OR audit OR accountability OR transparency OR risk OR management OR assessment OR cooperation OR treaty OR treaties) -is:retweet lang:en -decentralized -DAO -crypto -cryptocurrency -cryptocurrencies -#crypto -#cryptocurrency -L2 -#L2 -#Layer2"

# Fetch tweets
# tweets_df_rsa = fetch_tweets(query_rsa, count=4209)
# save_to_csv(tweets_df_rsa,f"fetched_tweets/tweets_rsa_{date_dmy}.csv")
# df_rsa = tweets_df_rsa.copy()

# Read saved tweets dataset
tweets_df_rsa = pd.read_csv("fetched_tweets/tweets_rsa_{??}.csv")
tweets_df_rsa.head()

# Process
process_df("rsa").head()
# TODO tweets_df_rsa["Stance_weighted_reach"].describe().T

# Manual tweaks specific to this dataset
"""
#df.loc[df["Tweet"].str.contains("firstever global ai summit last year")]
change_classification(tweets_df_rsa, "exploited consumer bluwhale leverage", "Safety")
change_classification(tweets_df_rsa, "partnering push boundary web3 ai ten offer l2", "Safety")
change_classification(tweets_df_rsa, "taylor swift lead heavy regulation ai", "Safety")
change_classification(tweets_df_rsa, "completely unacceptable rely ai translation manga industry fav", "Safety")
change_classification(tweets_df_rsa, "mam file fir handle sharing deepfake", "Safety")
change_classification(tweets_df_rsa, "surgical strike fake video coming soon minister", "Safety")
# TODO WHY WASNT THIS ABOVE LABELLED SAFETY? IT'S SO OBVIOUS
change_classification(tweets_df_rsa, "get twisted ai fascism fully buddied reason america", "Capabilities")
change_classification(tweets_df_rsa, "firstever global ai summit last year laid vision future", "Safety")
change_classification(tweets_df_rsa, "natural state free market deflation everything stop political ", "Capabilities")
change_classification(tweets_df_rsa, "hunger rt like pushed towards circulating ai generated", "Safety")
change_classification(tweets_df_rsa, "collaboration centralized exchange grok reveals copxdao", "Capabilities")
change_classification(tweets_df_rsa, "nearprotocol latest quill ilblackdragon cofounder", "Capabilities")
change_classification(tweets_df_rsa, "ai legislation tracker new ai law introduced worldwide", "Safety")
change_classification(tweets_df_rsa, "good morning amitshah ji goi use human ai generate", "Capabilities")
change_classification(tweets_df_rsa, "inconvenient truth scraping essential ai violates nearly core privacy", "Capabilities")
change_classification(tweets_df_rsa, "regulating ai paper wchristmarsden ai coregulation", "Safety")
change_classification(tweets_df_rsa, "china watching return policy old communist system known time", "Capabilities")
change_classification(tweets_df_rsa, "chat control 2 sommaruppdatering", "Safety")
change_classification(tweets_df_rsa, "im testifying u senate thursday ai need privacy law", "Safety")
change_classification(tweets_df_rsa, "marketplace improving efficiency transparency automates", "Safety")

recalculate_reclassification(tweets_df_rsa)
"""

tweets_df_rsa.groupby("Classification")["Stance_0to100"].agg(["count", "mean"])

# Final score of RSA
final_rsa = tweets_df_rsa["Stance_0to100"].mean() # 50.89
final_rsa = 50.89


## PUBLIC SENTIMENT ANALYSIS
query_psa = "AI (governance OR policy OR regulation OR capabilities) OR (#AIgovernance OR #AIpolicy) -is:retweet lang:en -decentralized -DAO -blockchain -#blockchain -blockchainassn -crypto -cryptocurrency -cryptocurrencies -#crypto -#cryptocurrency -L2 -#L2 -#Layer2"

# Fetch tweets
# tweets_df_psa = fetch_tweets(query_psa, count=500)
# save_to_csv(tweets_df_psa,f"fetched_tweets/tweets_psa_{date_dmy}.csv")
# df_psa = tweets_df_psa.copy()

# Read saved tweets dataset
tweets_df_psa = pd.read_csv("fetched_tweets/tweets_psa_{??}.csv")
tweets_df_psa.head()

# Process
process_df("psa").head()
# TODO tweets_df_psa["Stance_weighted_reach"].describe().T

# Manual tweaks specific to this dataset
"""
#df.loc[df["Tweet"].str.contains("firstever global ai summit last year")]
change_classification(tweets_df_psa, "cess_storage excited join", "Capabilities")
change_classification(tweets_df_psa, "firstever global ai summit last year", "Safety")
change_classification(tweets_df_psa, "let touch base rgt rgt governance token revoxs", "Capabilities")
change_classification(tweets_df_psa, "claude 35 sonnet transformed research paper", "Capabilities")
change_classification(tweets_df_psa, "ai face growing censorship", "Safety")
change_classification(tweets_df_psa, "zayaai brings future healthcare within", "Capabilities")
change_classification(tweets_df_psa, "web3 adoption skyrocketing aicrypto project boasting", "Capabilities")
change_classification(tweets_df_psa, "initial reaction overstated poking around morning", "Capabilities")
change_classification(tweets_df_psa, "ai theft stealing voice", "Capabilities")
change_classification(tweets_df_psa, "masa x gt protocol exciting news masa", "Capabilities")
change_classification(tweets_df_psa, "excited kick broadband commission working group data governance digital age data", "Safety")
change_classification(tweets_df_psa, "policymakers laid thoughtful framework regulation promote responsible", "Safety")
change_classification(tweets_df_psa, "malicious abuse ai example no2 vulnerability", "Capabilities")
change_classification(tweets_df_psa, "generative ai continues misused abused malicious individual", "Capabilities")
change_classification(tweets_df_psa, "first came apple product classified", "Capabilities")

recalculate_reclassification(tweets_df_psa)
"""

tweets_df_psa.groupby("Classification")["Stance_0to100"].agg(["count", "mean"])

# Final score of PSA
final_psa = tweets_df_psa["Stance_0to100"].mean() # 70.77
final_psa = 70.77


## INVESTMENTS
invcap = [95.14, 95.53, 95.39, 95.35, 97.02, 97.71, 97.67, 97.03, 95.93, 96.04, 96.03, 94.78, 92.94]
invsaf = [87.97, 86.03, 82.53, 81.31, 80.42, 75.87, 80.56, 85.30, 84.21, 88.92, 88.45, 88.18, 76.88]

final_incap = invcap[-1]
final_insaf = invsaf[-1]


## FINAL TOTAL CALCULATION
airi_score = round(0.25*(final_psa + final_rsa + final_incap + final_insaf), ndigits=2)


"""
### Little tests to check if it works
df["Tweet"].head()

sia = SentimentIntensityAnalyzer()
sia.polarity_scores("We're going to get more powerful AI systems, and they will inevitably diffuse more over time. Rather than just focusing on limiting capabilities, we should focus on adapting to their impacts.")
sia.polarity_scores("AI safety is very important, it deserves great investment and effort and I am excited to work in this field.")
sia.polarity_scores("AI capabilities is very important, it deserves great investment and effort and I am excited to work in this field.")
sia.polarity_scores("While AI safety is important, putting up regulations to ensure it limits human mind and maybe we should see what our smartest can do with AI.")

df2 = pd.DataFrame({
    "Tweet": [
        "Even if use finetuning, we cannot guarantee AI safety.",
        "Finetuning should work to ensure we can continue working on AI and make it more capable.",
        "'alignment', 'robustness', 'bias', 'fair', 'fairness', 'interpretability', 'explainability', 'transparency', 'transparent', 'accountability', 'oversight', 'trustworthiness', 'trustworthy', 'adversarial', 'mitigation', 'compliance', 'compliant', 'standard', 'audit', 'governance', 'best practice', 'ethical AI', 'AI principle', 'human-in-the-loop'",
        "'model drift', 'dataset bias', 'overfitting', 'underfitting', 'training data', 'validation data', 'fine-tuning', 'pre-training', 'prompt engineering', 'reinforcement learning', 'reward modeling', 'human feedback', 'differential privacy', 'federated learning', 'adversarial attacks', 'defense mechanism', 'robustness check', 'anomaly detection', 'safety protocol', 'kill switch', 'fail-safe', 'control system', 'constraint satisfaction', 'value alignment', 'scalable oversight', 'corrigibility', 'interpretability tool', 'post-hoc analysis', 'causality', 'counterfactual reason', 'predictive accuracy', 'performance metric'",
        "'AI policy', 'legislative framework', 'compliance standard', 'AI ethic board', 'regulatory compliance', 'risk assessment', 'safety standard', 'policy recommendation', 'AI oversight', 'international cooperation', 'AI treaty'",
        "Artificial intelligence safety is very important, it deserves great investment and effort and I am excited to work in this field.",
        "AI capabilities is very important, it deserves great investment and effort and I am excited to work in this field.",
        "While AI safety is important, putting up regulations to ensure it limits human mind and maybe we should see what our smartest can do with AI.",
        "It's good that technological innovations are being limited by AI policies.",
        "engineering learning robustness regulatory compliance Legislative frameworks, protocols counterfactual treaties compliance standards, AI ethics boards. Many AI Metrics are here."]})

def process_df2(dataframe, safety_keywords, capability_keywords):
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
    # TODO add line to accommodate retweet count bc I will pull once more without RTs but with just the RT numbers
    return dataframe

process_df2(df2, safety_keywords_psa, capability_keywords_psa).head()


########### VISUALISATION

# Calculating term frequencies
tf = df["Tweet"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False)

# Barplot
tf[tf["tf"] > 200].plot.bar(x="words", y="tf")
plt.show()

# Plot the distribution of stances
sns.countplot(x='Stance', data=tweets_df)
plt.title('Stance Analysis of AI Governance Tweets')
plt.xlabel('Stance')
plt.ylabel('Number of Tweets')
plt.show()

# Plot the distribution of sentiments
sns.countplot(x='Sentiment_Class', data=tweets_df)
plt.title('Sentiment Analysis of AI Governance Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()

# Key concerns (most common words in all)
all_tweets = ' '.join(tweets_df['Tweet'])
wordcloud = WordCloud(width=800, height=400, max_words=100).generate(all_tweets)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of AI Governance Tweets')
plt.show()

# Key concerns (most common words) for each stance
for stance in ['AI Capabilities', 'AI Safety']:
    stance_tweets = ' '.join(tweets_df[tweets_df['Stance'] == stance]['Tweet'])
    wordcloud = WordCloud(width=800, height=400, max_words=100).generate(stance_tweets)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud of {stance} Tweets')
    plt.show()
"""