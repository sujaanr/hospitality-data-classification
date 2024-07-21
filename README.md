# Hotel Booking Cancellation Prediction

## Project Overview

This project aims to predict hotel booking cancellations using machine learning techniques. The dataset includes various features related to hotel bookings, and the goal is to classify whether a booking will be canceled or not.

## Dataset

The dataset contains the following features:

- `no_of_adults`: Number of adults
- `no_of_children`: Number of children
- `no_of_weekend_nights`: Number of weekend nights
- `no_of_week_nights`: Number of week nights
- `type_of_meal_plan`: Type of meal plan
- `required_car_parking_space`: Whether a car parking space is required
- `room_type_reserved`: Type of room reserved
- `lead_time`: Lead time (days)
- `arrival_year`: Year of arrival
- `arrival_month`: Month of arrival
- `arrival_date`: Date of arrival
- `market_segment_type`: Market segment type
- `repeated_guest`: Whether the guest is a repeated guest
- `no_of_previous_cancellations`: Number of previous cancellations
- `no_of_previous_bookings_not_canceled`: Number of previous bookings not canceled
- `avg_price_per_room`: Average price per room
- `no_of_special_requests`: Number of special requests
- `total_nights`: Total nights (derived feature)
- `guest_count`: Guest count (derived feature)
- `booking_status`: Target variable (0: Not Canceled, 1: Canceled)

## Project Steps

1. **Data Loading and EDA**
    - Load the training and test datasets.
    - Perform exploratory data analysis (EDA) to understand the data.
    - Handle missing values using the `SimpleImputer`.
    - Create additional features such as `total_nights` and `guest_count`.

2. **Data Preprocessing**
    - Separate features and target variable.
    - Identify categorical and numerical columns.
    - Apply transformations:
        - Numerical data: Standard scaling and PCA.
        - Categorical data: One-hot encoding.
    - Bundle preprocessing steps using `ColumnTransformer`.

3. **Feature Selection**
    - Use `SelectKBest` with chi-square statistics to select the top 10 features.

4. **Modeling**
    - Define models: Logistic Regression, Random Forest, and Gradient Boosting.
    - Perform hyperparameter tuning using `RandomizedSearchCV`.
    - Save the best models for each algorithm.

5. **Ensemble Learning**
    - Combine the best models using a Voting Classifier with soft voting.

6. **Model Evaluation**
    - Evaluate individual models and the ensemble model on training data.
    - Metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC.
    - Visualizations: Classification report, Confusion matrix heatmap, ROC Curve.

7. **Prediction on Test Data**
    - Use the best model (Voting Classifier) to predict on the test data.
    - Save the predictions to `pred.csv`.

8. **Performance Analysis**
    - Analyze the performance of the model across different classes.

## How to Run the Project

1. Clone the repository or download the project files.
2. Ensure you have the necessary libraries installed:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3. Place the training and test datasets in the appropriate directory.
4. Run the script:
    ```bash
    python MachineLearningChallengeFinal.ipynb
    ```

## Project Structure

- `MachineLearningChallengeFinal.ipynb`: Main project script.
- `train.csv`: Training dataset.
- `test.csv`: Test dataset.
- `pred.csv`: Predictions on the test dataset.
- `a.csv`: Accuracy on the training data.

## Results

The project successfully predicts hotel booking cancellations with the following performance metrics (example results):

- **Voting Classifier Model Performance:**
    - Accuracy: 0.85
    - Precision: 0.80
    - Recall: 0.78
    - F1 Score: 0.79
    - ROC AUC: 0.88

Visualizations:
- Confusion Matrix
- ROC Curve
