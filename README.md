## Project Structure

1. **Introduction**
   - This project focuses on predicting apartment prices using regression analysis. It involves data preprocessing, exploratory data analysis, feature engineering, and building a decision tree regression model.

2. **Libraries Used**
   - The project utilizes the following libraries and modules:
     - NumPy
     - Pandas
     - Matplotlib
     - Seaborn
     - Scikit-learn
     - Geopy

3. **Loading the Raw Data**
   - The raw data is loaded from an Excel file using the Pandas library.
   ```python
   raw_data = pd.read_excel("C:\\Users\\user\\Downloads\\Apartments_Data.xlsx")

4. **Preprocessing**
   - **Exploring Descriptive Statistics:**
     - Descriptive statistics are examined to gain insights into the variables.
     - Two sets of descriptive statistics are presented: one for numerical variables and another for both numerical and categorical variables.
     ```python
     display(raw_data.describe(include='all'))
     display(raw_data.describe())
     ```

   - **Determining Variables of Interest:**
     - Calculating the distance from the center using latitude and longitude to identify potential factors affecting prices.
     ```python
     raw_data['latitude_center'] = 41.327953
     raw_data['longitude_center'] = 19.819025
     raw_data['distance_from_center'] = raw_data.apply(lambda x: geodesic((x['lat'], x['lon']),(x['latitude_center'], x['longitude_center'])).km, axis=1)
     ```

   - **Adding New Columns:**
     - Utilizing comments in the dataset to add new columns such as 'Parkim' and 'Ashensor' based on the presence of specific keywords.
     ```python
     raw_data['Parkim'] = raw_data['comments'].apply(hasParking)
     raw_data['Ashensor'] = raw_data['comments'].apply(hasElevator)
     ```

   - **Cleaning Data:**
     - Removing unwanted characters (e.g., '$') from the 'price' column and converting it to a float.
     ```python
     raw_data['price'] = raw_data['price'].replace('[^\d.]', '', regex=True)
     raw_data['price'] = raw_data['price'].astype(float)
     ```

   - **Handling Missing Values:**
     - Identifying and handling missing values in the dataset.
     ```python
     data.isnull().sum()
     data_no_mv = data.dropna(axis=0)
     ```

   - **Exploring PDFs and Identifying Outliers:**
     - Visualizing the probability distribution function (PDF) of the 'price' variable and addressing outliers.
     ```python
     sns.distplot(data_no_mv['price'])
     # Additional steps for handling outliers
     ```

5. **Exploratory Data Analysis**
   - Visualizations depicting relationships between variables, including scatter plots for price and different features.
   ```python
   f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(15, 3))
   ax1.scatter(data_cleaned['year'], data_cleaned['price'])
   # Additional scatter plots for other features
   plt.show()

6. **Create Dummy Variables**
   - Creating dummy variables from categorical data using the 'get_dummies' method.

7. **Regression Model**
   - Declaring inputs and targets for the regression model.
   - Scaling the data using StandardScaler.
   - Splitting the data into training and testing sets.
   - Building a decision tree regression model and evaluating its performance.
     ```python
     # Example code snippet
     model = DecisionTreeRegressor()
     model.fit(x_train, y_train)
     ```

8. **Model Evaluation**
   - Checking R-squared scores and residual plots to assess the model's goodness of fit.
     ```python
     # Example code snippet
     train_score = model.score(x_train, y_train)
     ```

9. **Testing**
   - Hyperparameter tuning using grid search for DecisionTreeRegressor.
     ```python
     # Example code snippet
     param_grid = {'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
     grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
     grid_search.fit(x_train, y_train)
     ```
   - Evaluating the model on the testing set.
     ```python
     # Example code snippet
     best_params = grid_search.best_params_
     best_model = DecisionTreeRegressor(**best_params)
     test_score = best_model.score(x_test, y_test)
     ```

10. **Results**
    - Visualizations of actual vs predicted prices on both the training and testing sets.
      ```python
      # Example code snippet
      plt.scatter(y_train, y_hat)
      plt.xlabel('Targets (y_train)', size=18)
      plt.ylabel('Predictions (y_hat)', size=18)
      plt.show()
      ```

## Improvements

**Note:** This project was developed for learning purposes and is not intended for production use. While it provides insights into regression analysis and decision tree modeling, there are several areas for improvement:

1. **Data Quality Enhancement**

2. **Feature Engineering**

3. **Model Optimization**

4. **Handling Categorical Data**

5. **Model Interpretability and Testing**
