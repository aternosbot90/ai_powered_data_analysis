# ai_powered_data_analysis
a python code to make the csv file data analysis by ai 

python code:-
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Step 1: Data Ingestion and Preprocessing
def load_data(file_path):
    data = pd.read_csv(file_path)
    st.write("Data Preview:")
    st.write(data.head())
    return data

def preprocess_data(data):
    # Handling missing values
    data = data.dropna()  # For simplicity, dropping missing values
    # Encoding categorical features, scaling numerical features
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = pd.factorize(data[column])[0]
    scaler = StandardScaler()
    data[data.columns] = scaler.fit_transform(data)
    return data

# Step 2: Feature Engineering (Example)
def feature_engineering(data):
    # Example: Adding interaction features
    if 'feature1' in data.columns and 'feature2' in data.columns:
        data['feature_interaction'] = data['feature1'] * data['feature2']
    return data

# Updated Step 3: Model Training and Selection
def train_model(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check if the target is continuous or categorical
    if y.nunique() > 10 or y.dtype == 'float':
        # Continuous target: use a regression model
        model = RandomForestRegressor()
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        # Evaluate the best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        st.write("Regression Model Evaluation:")
        st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        st.write("R-squared:", r2_score(y_test, y_pred))
    else:
        # Discrete target: use a classification model
        model = RandomForestClassifier()
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        # Evaluate the best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        st.write("Confusion Matrix:")
        plot_confusion_matrix(y_test, y_pred)
    
    return best_model

# Updated Confusion Matrix Plot Function
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Step 4: Visualization
def visualize_data(data):
    st.write("Data Correlation Heatmap:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

# Step 5: Real-Time Dashboard with Streamlit
def main():
    st.title("AI-Powered Data Analysis System")
    
    # Upload Dataset
    uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
    
    if uploaded_file is not None:
        # Load and preprocess data
        data = load_data(uploaded_file)
        data = preprocess_data(data)
        
        # Feature Engineering
        data = feature_engineering(data)
        
        # Separate features and target (assuming the target is in the last column)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        # Train and evaluate model
        model = train_model(X, y)
        
        # Visualize data insights
        visualize_data(data)
    
        st.write("Model is ready for predictions.")

# Entry point
if __name__ == "__main__":
    main()
