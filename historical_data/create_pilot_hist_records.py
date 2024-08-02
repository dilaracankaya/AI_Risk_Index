import pandas as pd
from datetime import datetime, timedelta
from warnings import filterwarnings
filterwarnings('ignore')

from indicators import hist_invsaf, hist_invcap, save_to_csv

hist_rsa = [40.12, 42.15, 43.67, 44.88, 46.20, 47.35, 48.58, 49.12, 49.87, 50.24, 50.67, 51.22, 50.89]
hist_psa = [62.34, 63.56, 65.12, 66.78, 67.89, 68.45, 68.90, 69.12, 69.45, 69.78, 70.12, 70.50, 70.77]


today = datetime.today()
if today.weekday() == 0:  # Monday is 0
    end_date = today
else:
    end_date = today - timedelta(days=today.weekday())

dates = [end_date - timedelta(weeks=i) for i in range(len(hist_invcap))]
date_labels = [date.strftime('%b %d') for date in reversed(dates)]


# Create DataFrames
investments_capabilities = pd.DataFrame({'Date': date_labels, 'invcap_Indicator_Score': hist_invcap})
save_to_csv(investments_capabilities,"historical_data/hist_invcap.csv")

investments_safety = pd.DataFrame({'Date': date_labels, 'invsaf_Indicator_Score': hist_invsaf})
save_to_csv(investments_safety,"historical_data/hist_invsaf.csv")

tweets_psa = pd.DataFrame({'Date': date_labels, 'psa_Indicator_Score': hist_psa})
save_to_csv(tweets_psa,"historical_data/hist_psa.csv")

tweets_rsa = pd.DataFrame({'Date': date_labels, 'rsa_Indicator_Score': hist_rsa})
save_to_csv(tweets_rsa,"historical_data/hist_rsa.csv")
