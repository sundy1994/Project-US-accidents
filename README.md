# Project: Analysis and Modeling of Traffic Accidents in US
**Key Words:**  Python, Data cleaning, Data wrangling, EDA, Data visualization, Machine Learning, Modeling


## Overview:

There are more than 30K motor vehicle crashes yearly in the US, which involves more than 3K deaths. In this project, I am interested in getting in-depth insights into traffic accidents in the US. We want to visualize the traffic accident data in various ways to present the US traffic accidents cases in a story-telling way to help audiences to learn about the situations of US traffic accidents. It is also interesting to understand the significant factors causing traffic accidents, which might increase our traffic safety awareness in the future.

The main purpose of this project it to study the key features influencing the occurrence of car accidents, the factors affecting accidents severity, as well as time and location that have the highest number of accidents. In order to do this, I used Python(Pandas) in Google Colab to clean the original dataset, complete EDA and implement data visualization and exhibit some interesting features on map. After that, I trained some machine learning models including Linear Regression, KNN, Decision Tree, Random Forest, SVM and Neural Network to Predicting the severity and duration of accidents. Most of them have testing errors above 86%, which is pretty good.


## Data Source: [A Countrywide Traffic Accident Dataset (2016 - 2021)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) from Kaggle

This dataset contains 1.5 million accident records, which is collected from February 2016 to Dec 2020 in 49 states of the USA. The author uses multiple APIs provided by the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road-networks.

## Part I: Data Cleaning and EDA

### Dataset Overview
The original dataset contains 1516064 entries with 46 attributes, including 13 bool values, 13 float values, 1 int value and 20 objects. They can be roughly devided into **Time** attributes, **Location** attributes and **Traffic Environment** attributes. After removing Null values and useless data, it ended up with 1370980 rows.

### Distribution of Severity
The severity of accidents is a number between 1 and 4, where 1 indicates the least impact on traffic and 4 indicates the most severe cases. Most of the accidents have severity 2 while severity 3 takes second place. This makes sense since, but the imbalance in the distribution of severity may influence the accuracy of prediction. Need to consider unsampling some severity 2 cases to improve the performance of our model.
![](https://github.com/sundy1994/Project-US-accidents/blob/main/images/1.%20severity.png)
