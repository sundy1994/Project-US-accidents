# Project: Analysis and Modeling of Traffic Accidents in US
**Key Words:**  Python, Data cleaning, Data wrangling, EDA, Data visualization, Machine Learning, Classification


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

### Location Analysis
A natural question to ask is which states are dangerous/safe in terms of traffic accidents? Based on this dataset, California has most accidents (about 30% of all cases) in US. Florida takes the second place but CA is much ahead of it. They also have many accidents with severity 4.

![](https://github.com/sundy1994/Project-US-accidents/blob/main/images/2.%20most%20states.png)

On the other hand, SD, WY, VT and ND only have less than 500 accidents over the past 5 years. That's really few. However, WY has the highest percentage of severity 4, so it should not be considered very safe.

![](https://github.com/sundy1994/Project-US-accidents/blob/main/images/3.%20least%20states.png)

As for cities, the numbers in LA, Miami and Orlando explaines the cases in CA and FL. After all, who can say no to Disneyland and Universal Studio?

![](https://github.com/sundy1994/Project-US-accidents/blob/main/images/4.%20most%20cities.png)

I further use Plotly to visualize cases of cities on an interactive US map (best open with Google Colab). Although several big cities on west coast have many cases, such as LA, Sacramento and Portland, obviously the situation on east coast is worse.

![](https://github.com/sundy1994/Project-US-accidents/blob/main/images/5.%20US%20map.png)

Finally, let's focus on the accidents in Philadelphia specifically where our university is located, and do visualization on 2 interactive maps (again, best open with Google Colab) to see which part of Philadelphia is more likely to have accidents. 

The first map shows a bunch of circles, in which orange means a large number of accidents, while green means small. If you keep clicking on the circles, you will finally got some warning signs, and these signs are the exact locations of accidents. The color of the warning signs means the severity, where the red means the most severe accidents, and green means least.

![](https://github.com/sundy1994/Project-US-accidents/blob/main/images/6.%20Philly%20map.png)

The second map is a density map, where the color means average severity of accidents. Yellow means the most severe and purple means the least severe. Collectively, the accidents mainly located in center city and east philly. There are lots of severe cases on I-676, I-76 and I-95. This analysis can help informing drivers about the dangerous road part.

![](https://github.com/sundy1994/Project-US-accidents/blob/main/images/7.%20Philly%20map%202.png)

### Time Analysis
When are the accidents most likely to happen? Or, happens most frequently? According to the data, the number of accidents is less in summer (July and August), and higher in winter (November and December). This might be caused by the worse weather conditions in winter.

![](https://github.com/sundy1994/Project-US-accidents/blob/main/images/8.%20month.png)

In terms of weekdays, we can see that the number of accidents in the workdays are twice of which in the weekends. This is because people needs to drive to work during rush hours so there are many cars appearing at the same time, and thus increase the chance of car accidents.

![](https://github.com/sundy1994/Project-US-accidents/blob/main/images/9.%20weekday.png)

The plot below confirms my hypothesis. There are three common peaks of acccidents in a day: 7-8 AM, 16-17 PM, 0 AM. 7-8 AM and 16-17 PM are rush hours, while 0 AM is a time when people are very tired driving (and a time when people usually leave resturants or head to bars and clubs), which will leads to accidents.

![](https://github.com/sundy1994/Project-US-accidents/blob/main/images/10.%20hour.png)

### Text Analysis
There's 'Description' column in the dataset, which contains the information of each accident. I first create a word cloud, which indicates some common causes of accidents such as 'lane blocked', 'slow traffic' and 'exit closed'.

![](https://github.com/sundy1994/Project-US-accidents/blob/main/images/11.%20text.png)

Finally, here are the most common words in the description. 79.75% of rows contains 'Accidents' or 'Incident', 41.02% of rows contains 'Road' or 'Rd' or 'Ave' or 'st', 29.42% of rows contains 'exit' and 9.94% of rows contains 'closed'.

![](https://github.com/sundy1994/Project-US-accidents/blob/main/images/12.%20text2.png)


## Part 2: Training and Prediction with Machine Learning Models

### Overview
The goal of the modeling part is predicting the severity of a traffic accident be based on the key features. Since severity has 4 levels, I mainly used classifiers to make predictions.

To make the dataset suitable for training with machine learning models, I first converted 12 categorical variables into labels using LabelEncoder from sklearn, and then manually dropped all the unnecessary features based on our domian knowledges (such as Zipcode and description). After that, I divided the dataset into features (41 columns in total) and label (severity) and split it into training set and testing set. As part of feature engineering, I use data mining techinques to extract features from raw, clean datasets based on our domain knowledges & common sense in order to improve the performance of machine learning model.

### Unsupervised models: PCA and K-means clustering
To begin with, unsupervised models including PCA and k-means clustering can help selecting features, reducing noise and solve multicolinearity problems. After standerdize the dataset with StandardScaler from sklearn, the training set is fited with PCA. Based on the explained variance, I selected 31 PCs to build new train_pca and test_pca.

K-means clustering seeks to find “natural groupings” in data based on distance. By plotting distortion vs. number of clusters, I use the "elbow method" to determine k = 4 as the number of clusters.

### Supervised models

**Naive Bayes Classifier**

I used GaussianNB from sklearn. Using training set without PCA, I get a testing accruacy of 79.0391%, while the train_pca gives a testing accuracy of 79.7007%. There isn't any big improvement, but there's only 31 columns in train_pca, which saves space comparing to 41 columns in the original training set.

**Decision Tree**

Decision Tree is imported from sklearn.tree.DecisionTreeClassifier(). Without pca, I received a testing accuracy of 84.7618% and 84.4178% with PCA. 

**Random Forest**

Random forest model is basicalky ensembles of decision tree. I imported RandomForestClassifier from sklearn.ensemble. With n_estimaors = 100, the testin accuracy accuracy is 86.8688%.

**Logistic Regression**

LogisticRegression was imported from sklearn.linear_model. This model generates an testing accuracy of 86.2758%. Both PCA and regularization can't improve the testing accuracy.

**Stacking**

Since logistic regression and random forest give better results, I want to whether the testing accuracy will be even higher if we stack them together. I imported stacking classifier from sklearn.ensemble. As what I expected, the combined model gives a testing accuracy of 89.1705%. This will be our final model.


## Summary
In this project, I explored the distribution of car accidents with respect to location and time. Then, I implemented some machine learning classifiers to predict the severity of car accidents. By stacking Random Forest and Logistic Regression, I obtained a testing accuracy of 89.1705%.
As a next step, I may analyze the wheather conditions and traffic environment, which wasn't focused in this project. It's easy to predict that traffic accidents tend to happen more frequently in rainy and snowy days, where the signs are less conspicuous. However, this need to be validated with more data in future analysis.
