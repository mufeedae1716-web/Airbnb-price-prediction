# 🏠 Airbnb Listing Price Prediction

##  Project Overview

This project predicts the price of Airbnb listings using Machine Learning regression techniques.

The dataset contains information about Airbnb properties such as bedrooms, beds, review scores, location, availability, minimum nights, maximum nights, and other listing-related features.

The main goal of this project is to build a Machine Learning model that can predict the price of an Airbnb listing based on its available features.

---

##  Problem Definition

Airbnb listing prices vary depending on different factors such as location, number of bedrooms, beds, reviews, availability, and minimum/maximum nights.

The objective of this project is to develop a Machine Learning regression model that can predict the price of an Airbnb listing using these features.

---

##  Objectives

- Analyze the Airbnb listing dataset.
- Perform Exploratory Data Analysis (EDA).
- Handle missing values and data inconsistencies.
- Perform feature selection.
- Encode categorical features.
- Scale numerical features where required.
- Train multiple regression models.
- Compare model performance.
- Select the best-performing model.
- Create a Machine Learning pipeline.
- Save the trained pipeline as a `.pkl` file.
- Deploy the model using Streamlit.

---

##  Dataset

The dataset contains Airbnb listing information.

### Important Features

- accommodates
- bedrooms
- beds
- review_scores_location
- reviews_per_month
- calculated_host_listings_count
- host_total_listings_count
- host_total_listings_count_shared_rooms
- number_of_reviews
- latitude
- longitude
- minimum_nights
- minimum_nights_avg_ntm
- maximum_minimum_nights
- maximum_nights
- calculated_host_listings_count_entire_homes
- number_of_reviews_ltm
- availability_90
- availability_365
- price

`price` is the target variable.

---

##  Exploratory Data Analysis

The following analysis was performed:

- Dataset shape and information
- Data type analysis
- Missing value checking
- Price distribution
- Outlier detection
- Correlation analysis
- Correlation heatmap
- Feature selection

---

##  Data Preprocessing

The following preprocessing steps were performed:

1. Removed unnecessary features such as `id`.
2. Checked and handled missing values.
3. Selected the top 20 features based on correlation with price.
4. Encoded categorical variables.
5. Applied numerical preprocessing.
6. Split the dataset into training and testing sets.
7. Applied feature scaling where required.

---

##  Machine Learning Models

Three regression models were trained and compared:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

###  Best Model

**Random Forest Regressor** was selected as the best-performing model based on the model evaluation results.

---

##  Model Evaluation

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The model with the best overall performance was selected for the final pipeline.

---

##  Machine Learning Pipeline

The final workflow is:

```text
Raw Data
   ↓
Preprocessing
   ↓
Encoding
   ↓
Scaling
   ↓
Random Forest Regressor
   ↓
Final Pipeline
   ↓
Save as .pkl
