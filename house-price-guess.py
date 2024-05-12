# by OkaySidarKarakaya (@saymygrace)

import pandas as pd  # Import the pandas library as pd to use its functionalities
import numpy as np  # Import the numpy library as np to use its functionalities
import matplotlib.pyplot as plt  # Import the pyplot module from matplotlib library as plt to create plots
import seaborn as sns  # Import seaborn library as sns to enhance visualizations

data = pd.read_csv("housing.csv")  # Read the dataset "housing.csv" and store it in the variable data
data.dropna(inplace=True)  # Drop rows with missing values from the dataset inplace

from sklearn.model_selection import train_test_split  # Import train_test_split function from sklearn.model_selection module

X = data.drop(["median_house_value"], axis=1)  # Extract features from the dataset excluding "median_house_value" column and store it in X
y = data["median_house_value"]  # Extract target variable "median_house_value" and store it in y

# Split the dataset into training and testing sets with 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_data = X_train.join(y_train)  # Combine X_train and y_train into a single DataFrame called train_data

# Convert categorical columns to dummy variables
train_data = pd.get_dummies(train_data, columns=["ocean_proximity"])

# Plot histograms for each column in train_data with a size of 15x8 inches
train_data.hist(figsize=(15, 8))

# Create a new figure with a size of 15x8 inches for the heatmap visualization
plt.figure(figsize=(15, 8))

# Plot a heatmap of the correlation matrix of train_data with annotations using a color map of YlGnBu
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")
plt.show()  # Show the plot

# Plot another heatmap for correlation matrix without displaying it
sns.heatmap(train_data.corr(),annot=True, cmap = "YlGnBu")  

# Apply log transformation to certain numerical features to reduce skewness
train_data["total_rooms"] = np.log(train_data["total_rooms"]+ 1)
train_data["total_bedrooms"] = np.log(train_data["total_bedrooms"]+ 1)
train_data["population"] = np.log(train_data["population"]+ 1)
train_data["house_holds"] = np.log(train_data["house_holds"]+ 1)

# Plot histograms for each column in train_data after log transformation
train_data.hist(figsize=(15, 8))

# Convert categorical columns to dummy variables again
train_data = pd.get_dummies(train_data, columns=["ocean_proximity"])
train_data.drop(["ocean_proximity_<1H OCEAN"], axis=1, inplace=True)

# Create a scatter plot of latitude vs longitude with hue representing median_house_value using a coolwarm color palette
plt.figure(figsize=(15 ,8))
sns.scatterplot(x="latitude",y="longitude",data=train_data, hue="median_house_value",palette="coolwarm")
train_data["bedroom_ratio"] = train_data["total_bedrooms"] / train_data["total_rooms"]
train_data["household_rooms"] = train_data["total_rooms"] / train_data["households"]

# Create a new figure with a size of 15x8 inches for the heatmap visualization
plt.figure(figsize=(15, 8))

# Plot a heatmap of the correlation matrix of train_data with annotations using a color map of YlGnBu
sns.heatmap(train_data.corr(),annot=True, cmap = "YlGnBu")

from sklearn.linear_model import LinearRegression# Import Linear Regression model from sklearn.linear_model module
from sklearn.preprocessing import StandardScaler  # Import StandardScaler from sklearn.preprocessing module

scaler = StandardScaler() # Initialize StandardScaler object to scale features

X_train,y_train = train_data.drop(["median_house_value"], axis = 1),train_data["median_house_value"]
X_train_s = scaler.fit_transform(X_train) # Standardize the training features using StandardScaler


reg= LinearRegression() # Initialize Linear Regression model
reg.fit(X_train_s,y_train) # Train the Linear Regression model on the standardized training data
test_data = X_test.join(y_test) # Combine X_test and y_test into a single DataFrame called test_data

# Apply the same transformations to test_data as done on train_data
test_data ["total_rooms"] = np.log(test_data ["total_rooms"]+ 1) 
test_data ["total_bedrooms"] = np.log(test_data ["total_bedrooms"]+ 1)
test_data ["population"] = np.log(test_data ["population"]+ 1)
test_data ["house_holds"] = np.log(test_data ["house_holds"]+ 1)
test_data = pd.get_dummies(test_data, columns=["ocean_proximity"])
test_data.drop(["ocean_proximity_<1H OCEAN"], axis=1, inplace=True)

# Calculate the bedroom_ratio and household_rooms features for test_data
test_data ["bedroom_ratio"] = test_data ["total_bedrooms"] / test_data ["total_rooms"]
test_data ["household_rooms"] = test_data ["total_rooms"] / test_data ["households"]
X_test,y_test = test_data.drop(["median_house_value"], axis = 1),test_data["median_house_value"]

# Standardize the testing features using the same StandardScaler object
X_test_s = scaler.transform(X_test)

# Evaluate the performance of the Linear Regression model on the standardized testing data
reg.score(X_test_s,y_test)
from sklearn.ensemble import RandomForestRegressor # Import RandomForestRegressor model from sklearn.ensemble module

forest = RandomForestRegressor() # Initialize RandomForestRegressor model

# Train the RandomForestRegressor model on the standardized training data
forest.fit(X_train_s,y_train) 


# Evaluate the performance of the RandomForestRegressor model on the standardized testing data
forest.score(X_test_s,y_test)

from sklearn.model_selection import GridSearchCV # Import GridSearchCV from sklearn.model_selection module
forest = RandomForestRegressor() # Re-initialize RandomForestRegressor model

param_grid = {  # Define a dictionary containing hyperparameters for tuning
    "n_estimators" : [100,200,300],
    "min_samples_split": [2,4],
    "max_depth": [None,4,8]
}


# Initialize GridSearchCV with RandomForestRegressor, parameter grid, 5-fold cross-validation, and scoring method
grid_search = GridSearchCV(forest,param_grid, cv=5,
                          scoring="neg_mean_squared_error",
                          return_train_score=True)

# Fit the GridSearchCV object to the standardized training data
grid_search.fit(X_train_s,y_train)

grid_search.best_estimator_ # Use the GridSearchCV object to perform hyperparameter tuning and find the best estimator
grid_search.best_estimator_.score(X_test_s,y_test) # Evaluate the performance of the best estimator on the standardized testing data

