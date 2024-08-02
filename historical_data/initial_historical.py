from common import *

hist_invcap = [95.39, 95.35, 97.02, 97.71, 97.67, 97.03, 95.93, 96.04, 96.03, 94.96, 94.84, 94.01, 95.29, 94.44]
hist_invsaf = [17.47, 18.69, 19.58, 24.13, 19.44, 14.70, 15.79, 11.08, 11.55, 11.57, 21.16, 30.41, 30.76, 26.54]


today = datetime.today()
if today.weekday() == 0:  # Monday is 0
    end_date = today
else:
    end_date = today - timedelta(days=today.weekday())

dates = [end_date - timedelta(weeks=i) for i in range(len(hist_invcap))]
date_labels = [date.strftime('%b %d') for date in reversed(dates)]


# Create DataFrames
investments_capabilities = pd.DataFrame({'Date': date_labels, 'Investment_Amount_(3-Week_SMA)': hist_invcap})
save_to_csv(investments_capabilities,"historical_data/hist_inv_cap.csv")

investments_safety = pd.DataFrame({'Date': date_labels, 'Investment_Amount_(3-Week_SMA)': hist_invsaf})
save_to_csv(investments_safety,"historical_data/hist_inv_saf.csv")
