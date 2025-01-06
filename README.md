# Bitcoin Price Prediction by the Minute

This project predicts the direction of Bitcoin's price movement at the minute level using machine learning models. The script implements Logistic Regression, Decision Tree, and Random Forest to classify whether the next price movement will be upward or downward.

---

## **Key Components**

### **1. Data Preparation**
#### Function: `load_and_prepare_data(file_path)`
- **Purpose:**
  - Reads Bitcoin minute-level price data from a CSV file.
  - Computes percentage changes in closing price (`Current Minute Return`) as the primary feature.
  - Generates lagged features (`Lag 1` to `Lag 5`) to capture temporal dependencies.
  - Creates a binary target variable, `Direction`, where 1 indicates an upward movement and 0 indicates a downward movement.
  - Adds a constant term for Logistic Regression models.

- **Mathematical Context:**
  - **Stationarity**: The script uses percentage changes to transform non-stationary price data into a potentially stationary form, crucial for time-series modeling.
  - **Lagged Features**: Represent autoregressive components commonly used in forecasting.

---

### **2. Feature Engineering**
#### Function: `add_rsi_zscore(df, periods=14)`
- **Purpose:**
  - Calculates the **Relative Strength Index (RSI)**, a technical indicator to identify overbought or oversold conditions.
  - Computes the **Z-Score** of `Lag 1`, which standardizes the data for better model performance.

- **Mathematical Context:**
  - **RSI Formula**:
    
$$
\text{RSI} = 100 - \frac{100}{1 + \frac{\text{Avg Gain}}{\text{Avg Loss}}}
$$


where average gains and losses are calculated using an exponential moving average (EMA).
  
  - **Z-Score**:

$$
z = \frac{x - \mu}{\sigma}
$$

where $$\( x \)$$ is a data point, $$\( \mu \)$$ is the mean, and $$\( \sigma \)$$ is the standard deviation. This normalization improves model performance.

---

### **3. Logistic Regression Model**
#### Function: `train_logistic_regression(df)`
- **Purpose:**
  - Trains a logistic regression model using `Lag 1` and a constant term.
  - Splits the data into training and testing sets.
  - Predicts the direction of Bitcoin price movement for the test set.

- **Mathematical Context:**
  - **Logistic Regression**:

$$
P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \dots + \beta_nX_n)}}
$$
    
Models the probability of the target variable as a linear function of the features, transformed by a sigmoid function.

---

### **4. Decision Tree Model**
#### Function: `train_decision_tree(df)`
- **Purpose:**
  - Trains a decision tree classifier with features `Lag 1`, `RSI`, and `z-score`.
  - Visualizes the decision tree and ranks features by importance.

- **Mathematical Context:**
  - **Tree Splitting**: Decision trees split data to minimize impurity, calculated using the Gini Index:

$$
G = 1 - \sum_{i=1}^k P_i^2
$$

    
where $$\( P_i \)$$ is the proportion of samples belonging to class $$\( i \)$$.

---

### **5. Random Forest Model**
#### Function: `train_random_forest(df)`
- **Purpose:**
  - Trains a Random Forest classifier to predict price direction.
  - Evaluates predictions on the test set.

- **Mathematical Context:**
  - **Ensemble Learning**: Combines multiple weak learners (decision trees) to create a robust model.
  - **Bootstrap Aggregation (Bagging)**: Randomly samples data and features to reduce overfitting and improve generalization.

---

### **6. Model Evaluation**
#### Function: `evaluate_model(y_test, y_pred, title)`
- **Purpose:**
  - Computes accuracy and confusion matrix for each model.
  - Visualizes the confusion matrix as a heatmap.

- **Mathematical Context:**
  - **Confusion Matrix**: Evaluates classification performance by showing true positives, false positives, true negatives, and false negatives.
  - **Accuracy**:

$$
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
$$

---

### **7. Predictive Insights**
#### Function: `predict_next_minute(model, lag1)`
- **Purpose:**
  - Predicts the price direction for the next minute using the trained logistic regression model and a user-provided `Lag 1` value.

- **Mathematical Context:**
  - Applies the learned logistic regression coefficients to predict the next minute’s price movement based on historical data.

---

## **Overall Workflow**
1. Load and preprocess the data to extract lagged features and calculate percentage changes.
2. Engineer additional features, including RSI and Z-Score, for improved model performance.
3. Train and evaluate three models: Logistic Regression, Decision Tree, and Random Forest.
4. Visualize results and analyze feature importance to understand the key drivers of predictions.
5. Allow real-time predictions using user-provided input for `Lag 1`.

---

## **Visualization**
- Confusion matrices are plotted for each model to assess performance.
- A decision tree is visualized to understand its structure and decision-making process.

---

## **How to Run**
1. Ensure `bitcoin_data.csv` is present in the same directory as the script.
2. Install the required Python libraries:
   ```bash
   pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn
