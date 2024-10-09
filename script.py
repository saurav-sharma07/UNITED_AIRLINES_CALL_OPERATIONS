import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Long AHT And AST Analysis

calls_df = pd.read_csv('./calls.csv', parse_dates=['call_start_datetime', 'agent_assigned_datetime', 'call_end_datetime'])
customers_df = pd.read_csv('./customers.csv')
reason_df = pd.read_csv('./reason.csv')
sentiment_df = pd.read_csv('./sentiment_statistics.csv')
calls_cleaned_df=calls_df.dropna()

reason_df['primary_call_reason'] = reason_df['primary_call_reason'].str.strip()  # Trim whitespace
reason_df['primary_call_reason'] = reason_df['primary_call_reason'].str.replace(r'\s+', ' ', regex=True)

replacements = {
    'Check In': 'Check-In',
    'Check-In': 'Check-In',
    'Post Flight': 'Post-Flight',
    'Post-Flight': 'Post-Flight',
    'Products & Services': 'Products and Services',
    'Products and Services': 'Products and Services',
}

for key, value in replacements.items():
    reason_df['primary_call_reason'] = reason_df['primary_call_reason'].str.replace(key, value, case=False)

merged_df = pd.merge(calls_df, reason_df, on='call_id', how='left')
merged_df = pd.merge(merged_df, sentiment_df, on=['call_id', 'agent_id'], how='left')

merged_df['call_time'] = (merged_df['call_end_datetime'] - merged_df['call_start_datetime']).dt.total_seconds()/60
print("maximum call length=",merged_df['call_time'].max())

merged_df['AHT'] = (merged_df['call_end_datetime'] - merged_df['agent_assigned_datetime']).dt.total_seconds()/60
 

merged_df['AST'] = (merged_df['agent_assigned_datetime'] - merged_df['call_start_datetime']).dt.total_seconds()/60


Q1 = merged_df['call_time'].quantile(0.1)
Q3 = merged_df['call_time'].quantile(0.95)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

merged_df = merged_df[(merged_df['call_time'] >= lower_bound) & (merged_df['call_time'] <= upper_bound)]


avg_call_time = merged_df['call_time'].mean()
avg_AHT = merged_df['AHT'].mean()
avg_AST = merged_df['AST'].mean()

print(f"Average Call time: {avg_call_time:.2f} minutes")
print(f"Average AHT: {avg_AHT:.2f} minutes")
print(f"Average AST: {avg_AST:.2f} minutes")

aht_by_reason = merged_df.groupby('primary_call_reason')['AHT'].mean().reset_index()

top_aht_reasons = aht_by_reason.sort_values(by='AHT', ascending=False).head(10)

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x='primary_call_reason', y='AHT', data=top_aht_reasons, palette='viridis')
plt.title('Top 10 Call Reasons with Highest Average Handle Time (AHT)')
plt.ylabel('Average Handle Time (minutes)')
plt.xlabel('Call Reason')
plt.xticks(rotation=30, ha='right')  
plt.tight_layout()  
plt.show()

call_reason_counts = merged_df['primary_call_reason'].value_counts()

most_frequent_reason = call_reason_counts.idxmax()
least_frequent_reason = call_reason_counts.idxmin()

aht_most_frequent = aht_by_reason[aht_by_reason['primary_call_reason'] == most_frequent_reason]['AHT'].values[0]
aht_least_frequent = aht_by_reason[aht_by_reason['primary_call_reason'] == least_frequent_reason]['AHT'].values[0]

percentage_difference = ((aht_most_frequent - aht_least_frequent) / aht_least_frequent) * 100

print(f"Percentage Difference in AHT between most frequent ({most_frequent_reason}) and least frequent ({least_frequent_reason}) call reasons: {percentage_difference:.2f}%")

merged_df['day_of_week'] = merged_df['call_start_datetime'].dt.day_name()

summary_by_day = merged_df.groupby('day_of_week').agg(
    Call_Volume=('call_id', 'count'),
    Average_AHT=('AHT', 'mean'),
    Average_AST=('AST', 'mean')
).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

print(summary_by_day)

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.bar(summary_by_day.index, summary_by_day['Call_Volume'], color='purple', alpha=0.6, label='Call Volume')
ax1.set_ylabel('Call Volume', color='purple')
ax1.tick_params(axis='y', labelcolor='purple')

ax2 = ax1.twinx()
ax2.plot(summary_by_day.index, summary_by_day['Average_AHT'], color='blue', marker='o', label='Average AHT', linewidth=2)
ax2.set_ylabel('Average AHT (minutes)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  
ax3.plot(summary_by_day.index, summary_by_day['Average_AST'], color='red', marker='s', label='Average AST', linewidth=2)
ax3.set_ylabel('Average AST (minutes)', color='red')
ax3.tick_params(axis='y', labelcolor='red')

plt.title('Call Volume, Average AHT, and Average AST by Day of the Week')
fig.tight_layout()  
ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=False, labelspacing=1.2)
ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.8), frameon=False, labelspacing=1.2)
ax3.legend(loc='upper left', bbox_to_anchor=(0, 0.6), frameon=False, labelspacing=1.2)

plt.show()

merged_df['hour_of_day'] = merged_df['call_start_datetime'].dt.hour

call_volume_by_hour = merged_df.groupby('hour_of_day').size()


high_volume_threshold = call_volume_by_hour.quantile(0.8)
high_volume_hours = call_volume_by_hour[call_volume_by_hour >= high_volume_threshold].index


merged_df['high_volume'] = merged_df['hour_of_day'].apply(lambda x: 1 if x in high_volume_hours else 0)
high_volume_aht = merged_df[merged_df['high_volume'] == 1]['AHT'].mean()
low_volume_aht = merged_df[merged_df['high_volume'] == 0]['AHT'].mean()

print(f"Average AHT during high-volume periods: {high_volume_aht:.2f} minutes")
print(f"Average AHT during low-volume periods: {low_volume_aht:.2f} minutes")


high_aht_threshold = merged_df['AHT'].quantile(0.8)
high_aht_calls = merged_df[merged_df['AHT'] >= high_aht_threshold]

high_aht_sentiment = high_aht_calls['average_sentiment'].mean()
low_aht_sentiment = merged_df[merged_df['AHT'] < high_aht_threshold]['average_sentiment'].mean()

high_aht_silence = high_aht_calls['silence_percent_average'].mean()
low_aht_silence = merged_df[merged_df['AHT'] < high_aht_threshold]['silence_percent_average'].mean()

print(f"Average Sentiment for high AHT calls: {high_aht_sentiment:.2f}")
print(f"Average Sentiment for low AHT calls: {low_aht_sentiment:.2f}")
print(f"Average Silence % for high AHT calls: {high_aht_silence*100:.2f}%")
print(f"Average Silence % for low AHT calls: {low_aht_silence*100:.2f}%")


customer_tone_analysis = merged_df.groupby('customer_tone')[['AHT', 'AST']].mean().reset_index()
agent_tone_analysis = merged_df.groupby('agent_tone')[['AHT', 'AST']].mean().reset_index()

print("Average AHT and AST by Customer Tone:")
print(customer_tone_analysis)

print("\nAverage AHT and AST by Agent Tone:")
print(agent_tone_analysis)

plt.figure(figsize=(10,6))
sns.barplot(x='customer_tone', y='AHT', data=customer_tone_analysis)
plt.title('Average AHT by Customer Tone')
plt.ylabel('Average AHT (minutes)')
plt.xlabel('Customer Tone')
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x='customer_tone', y='AST', data=customer_tone_analysis)
plt.title('Average AST by Customer Tone')
plt.ylabel('Average AST (minutes)')
plt.xlabel('Customer Tone')
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x='agent_tone', y='AHT', data=agent_tone_analysis)
plt.title('Average AHT by Agent Tone')
plt.ylabel('Average AHT (minutes)')
plt.xlabel('Agent Tone')
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x='agent_tone', y='AST', data=agent_tone_analysis)
plt.title('Average AST by Agent Tone')
plt.ylabel('Average AST (minutes)')
plt.xlabel('Agent Tone')
plt.show()

tone_combination_analysis = merged_df.groupby(['customer_tone', 'agent_tone'])[['AHT', 'AST']].mean().reset_index()

print("Average AHT and AST by Customer Tone and Agent Tone:")
print(tone_combination_analysis)

plt.figure(figsize=(12, 8))
pivot_aht = tone_combination_analysis.pivot(index="customer_tone", columns="agent_tone", values="AHT")
sns.heatmap(pivot_aht, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title('Heatmap of Average AHT by Customer Tone and Agent Tone')
plt.xlabel('Agent Tone')
plt.ylabel('Customer Tone')
plt.show()

plt.figure(figsize=(12, 8))
pivot_ast = tone_combination_analysis.pivot(index="customer_tone", columns="agent_tone", values="AST")
sns.heatmap(pivot_ast, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title('Heatmap of Average AST by Customer Tone and Agent Tone')
plt.xlabel('Agent Tone')
plt.ylabel('Customer Tone')
plt.show()


# Reducing Escalations to Agents

vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
transcripts_matrix = vectorizer.fit_transform(merged_df['call_transcript'].dropna())

lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_topics = lda_model.fit_transform(transcripts_matrix)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
feature_names = vectorizer.get_feature_names_out()
display_topics(lda_model, feature_names, no_top_words)



# Primary Call Reasons- Categorization and Prediction

df_cleaned = pd.concat([merged_df[['AHT', 'AST', 'average_sentiment', 'silence_percent_average']], merged_df['primary_call_reason']], axis=1)
df_cleaned = df_cleaned.dropna()
X = df_cleaned[['AHT', 'AST', 'average_sentiment', 'silence_percent_average']]
y = df_cleaned['primary_call_reason']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

test_df = pd.read_csv('./test.csv')
test_features = pd.merge(test_df, merged_df, on='call_id', how='left')
test_features_clean = test_features[['call_id', 'AHT', 'AST', 'average_sentiment', 'silence_percent_average']].dropna().copy()
X_test_final = test_features_clean[['AHT', 'AST', 'average_sentiment', 'silence_percent_average']]
test_predictions = clf.predict(X_test_final)
test_features_clean['primary_call_reason'] = test_predictions
test_final = pd.merge(test_df, test_features_clean[['call_id', 'primary_call_reason']], on='call_id', how='left')
test_final[['call_id', 'primary_call_reason']].to_csv('test_saurav_vikrant.csv', index=False)
print("Predictions saved to 'test_saurav_vikrant.csv'")
