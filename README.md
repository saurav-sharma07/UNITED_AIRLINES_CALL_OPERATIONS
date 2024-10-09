## United Airlines Call Center AHT & AST Analysis

### Project Overview
This project analyzes key call center metrics such as Average Handle Time (AHT) and Average Speed to Answer (AST) for a dataset from a customer support center. The focus is to identify key drivers of long call durations and to recommend strategies for improving efficiency and customer satisfaction.

### Data Files
- `calls.csv`: Contains details about individual call sessions including start and end times.
- `customers.csv`: Information about customers associated with the calls.
- `reason.csv`: The reasons for the calls categorized into primary reasons.
- `sentiment_statistics.csv`: Contains sentiment analysis data for each call, capturing both customer and agent sentiment.
- `test.csv`: A test dataset for prediction.

### Key Metrics
- **AHT (Average Handle Time)**: Time spent by agents handling calls.
- **AST (Average Speed to Answer)**: Time taken to assign an agent to a call.
- **Call Volume**: Number of calls during specific time intervals or categories.
- **Sentiment**: Captures customer and agent sentiment during the calls.
- **Silence Percentage**: Percentage of time the call was silent.

### Steps and Methodology

1. **Data Preprocessing**:
   - Cleaned and merged datasets (`calls.csv`, `reason.csv`, `sentiment_statistics.csv`) to ensure consistency.
   - Created new features such as `call_time`, `AHT`, `AST`, and applied filtering to remove outliers.

2. **Analysis of Call Durations**:
   - Calculated average values for call time, AHT, and AST.
   - Explored top call reasons by AHT, focusing on primary drivers of long handling times.
   - Analyzed the impact of customer and agent tones (polite, neutral, angry) on call durations.

3. **Visualizations**:
   - **Bar Plots**: Visualized top 10 call reasons with the highest AHT.
   - **Heatmaps**: Displayed the relationship between customer and agent tones and their effect on AHT and AST.
   - **Day of the Week Analysis**: Explored call volume, AHT, and AST trends by the day of the week.

4. **Modeling**:
   - Used **Latent Dirichlet Allocation (LDA)** for topic modeling from call transcripts to categorize common themes.
   - Applied a **Random Forest Classifier** to predict primary call reasons based on AHT, AST, sentiment, and silence percentages.

5. **Prediction**:
   - Predictions of primary call reasons for the test dataset (`test.csv`) were saved in `test_saurav_vikrant.csv`.

### How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/saurav-sharma07/United_airlines_call_operations.git
   ```
   
2. Install the required libraries:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn statsmodels
   ```
   
3. Ensure that the data files are in the project directory. (NOTE: calls.csv file is not uploaded to the repository due to file size. Therefore make sure to add call.csv file before running the code.)

4. Run the script:
   ```bash
   python script.py
   ```

5. View the predictions in the `test_saurav_vikrant.csv` file generated.

### Libraries Used
- `pandas`: Data manipulation and analysis.
- `matplotlib` & `seaborn`: Data visualization.
- `scikit-learn`: Machine learning models and classification metrics.
- `statsmodels`: Statistical modeling.
- `LatentDirichletAllocation` (LDA): Topic modeling for call transcripts.

### Key Insights
- **Tone Analysis**: Polite agent tones reduce AHT and improve customer satisfaction, while neutral or frustrated tones tend to increase handling time.
- **Peak Hours**: High AHT was found during high-volume periods, suggesting potential inefficiencies in resource allocation.
- **Call Reasons**: Certain call reasons, such as Mileage Plus and Checkout, had significantly higher AHT, indicating potential process improvements needed.

### Future Work
- **Further Automation**: Apply more advanced NLP techniques to extract deeper insights from call transcripts.
- **Self-Service Tools**: Identify call reasons that can be redirected to self-service options to further reduce AHT and AST.
- **Agent Support**: Implement real-time feedback for agents using AI-based tools.

### Author
This project was developed by **Saurav Sharma & Vikrant**.

---


