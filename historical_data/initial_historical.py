from common import *

invcap = [95.14, 95.53, 95.39, 95.35, 97.02, 97.71, 97.67, 97.03, 95.93, 96.04, 96.03, 94.78, 92.94]
invsaf = [87.97, 86.03, 82.53, 81.31, 80.42, 75.87, 80.56, 85.30, 84.21, 88.92, 88.45, 88.18, 76.88]


today = datetime.today()
if today.weekday() == 0:  # Monday is 0
    end_date = today
else:
    end_date = today - timedelta(days=today.weekday())

dates = [end_date - timedelta(weeks=i) for i in range(len(invcap))]
date_labels = [date.strftime('%b %d') for date in reversed(dates)]


# Create DataFrames
investments_capabilities = pd.DataFrame({'Date': date_labels, 'Investment_Amount_(3-Week_SMA)': invcap})
save_to_csv(investments_capabilities,"historical_data/hist_inv_cap.csv")

investments_safety = pd.DataFrame({'Date': date_labels, 'Investment_Amount_(3-Week_SMA)': invsaf})
save_to_csv(investments_safety,"historical_data/hist_inv_saf.csv")
