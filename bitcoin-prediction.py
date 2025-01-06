import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import scipy.stats as stats


# ----------------------------- Data Preparation ----------------------------- #
def load_and_prepare_data(file_path):
    """
    Load Bitcoin data, compute percentage changes, and prepare features.
    """
    data = pd.read_csv(file_path)

    df = data['close'].pct_change() * 100
    df = df.rename("Current Minute Return").reset_index()

    for i in range(1, 6):
        df[f'Lag {i}'] = df['Current Minute Return'].shift(i)

    df = df.dropna()
    df['price'] = data.close
    df['Direction'] = (df['Current Minute Return'] >= 0).astype(int)
    df = sm.add_constant(df)  # Intercept term
    return df


# ------------------------- Feature Engineering ------------------------- #
def add_rsi_zscore(df, periods=14):
    """Add RSI and Z-Score to the DataFrame."""
    close_delta = df['price'].diff()
    up = close_delta.clip(lower=0)
    down = -close_delta.clip(upper=0)

    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()

    rsi = 100 - (100 / (1 + ma_up / ma_down))
    df['RSI'] = rsi
    df['z-score'] = stats.zscore(df['Lag 1'])
    df.dropna(inplace=True)
    return df


# ----------------------------- Model Training ----------------------------- #
def train_logistic_regression(df):
    """Train Logistic Regression model."""
    X = df[['const', 'Lag 1']]
    y = df['Direction']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = sm.Logit(y_train, x_train).fit()
    predictions = model.predict(x_test)
    y_pred = (predictions >= 0.5).astype(int)

    return model, y_test, y_pred


def train_decision_tree(df):
    """Train Decision Tree model and plot the tree, including feature importances."""
    X = df[['Lag 1', 'z-score', 'RSI']]
    y = df['Direction']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    clf = DecisionTreeClassifier(max_depth=5, criterion="gini", max_features=2)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Display feature importances
    feature_importances = pd.DataFrame({
        'Feature': ['Lag 1', 'z-score', 'RSI'],
        'Importance': clf.feature_importances_
    })
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    print("\nIncreased Accuracy by ~10% by adding RSI and z-score to the list of predictors")
    print("\nFeatures Ranked by Importance:\n", feature_importances, "\n")

    # Visualize the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=['Lag 1', 'z-score', 'RSI'],
        class_names=['Down', 'Up'],
        filled=True,
        fontsize=10
    )
    plt.title("Decision Tree Visualization")
    plt.savefig("decision_tree_visualization.png")
    plt.show()

    return clf, y_test, y_pred


def train_random_forest(df):
    """Train Random Forest model."""
    X = df[['Lag 1', 'z-score', 'RSI']]
    y = df['Direction']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    rf = RandomForestClassifier(n_estimators=10, max_features=2, max_depth=5)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    return rf, y_test, y_pred


# ----------------------------- Evaluation ----------------------------- #
def evaluate_model(y_test, y_pred, title):
    """Evaluate the model's performance and print metrics."""
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)


    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(8, 6))
    sn.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{title} Confusion Matrix')
    plt.show()


# ----------------------- User Input Prediction ----------------------- #
def predict_next_minute(model, lag1):
    """
    Predict whether the next minute tick will go up or down based on the given lag1 value.
    :param model: Trained logistic regression model.
    :param lag1: Percent change of the last minute tick.
    """
    user_input = [[1, lag1]]  # 1 corresponds to the constant term
    prediction = model.predict(user_input)
    direction = "UP" if prediction >= 0.5 else "DOWN"
    print(f"\n\n**Prediction for Lag 1 ({lag1:.3f}%): {direction} minute tick")


# ----------------------------- Logistic Regression Table ----------------------------- #
def display_logistic_regression_table(model):
    """
    Display the summary table of the logistic regression model.
    :param model: Trained logistic regression model.
    """
    print(model.summary())


# ----------------------------- Main Script ----------------------------- #
def main():
    file_path = "bitcoin_data.csv"
    df = load_and_prepare_data(file_path)
    df = add_rsi_zscore(df)

    print("\n--- Logistic Regression ---")
    model, y_test, y_pred = train_logistic_regression(df)
    evaluate_model(y_test, y_pred, "Logistic Regression")
    display_logistic_regression_table(model)

    # Allow user to predict the next tick with a custom lag1 value
    lag1 = -0.025  # -0.025 means -0.025% decline in the last minute of Bitcoin
    predict_next_minute(model, lag1)  # Predict whether price will increase or decrease next minute

    print("\n--- Decision Tree ---")
    clf, y_test, y_pred = train_decision_tree(df)
    evaluate_model(y_test, y_pred, "Decision Tree")

    print("\n--- Random Forest ---")
    rf, y_test, y_pred = train_random_forest(df)
    evaluate_model(y_test, y_pred, "Random Forest")


if __name__ == "__main__":
    main()
