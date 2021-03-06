## Introduction

### 1.1 Background
  According to https://www.driverknowledge.com/car-accident-statistics/, there are 6 million car accidents on average a year. Of those, 3 million experience injuries, 2 million of which are permanent injuries. How can one know if the accident they may possibly be in is going to leave them with an no injury, injury, serious injury, or death? This is the goal of this project.

### 1.2 Business Problem
  The problem presented is how to predict how severe a car crash is given the conditions it occurs in.  We are trying to inform people of the severity of a car accident under certain conditions. This information can help people make their own judgements about if they want to go down a specific way or wait for better conditions. With this information, the number of car accidents resulting in injury or worse may be reduced.

### 1.3 People of Interest
  This is useful to the general public because with this information they can decide when and where they are going to travel to minimize severity of car accidents if they do get in one. 

## The Data

### 2.1 The Source 
  The data I am using is the data given in this course, which is for Seattle and the timeframe this data was gathered in is 2004 - Present. 

  There are 37 features we can utilize to help predict how severe a car crash will be. There are also 194673 observations we can learn from. The data includes features such as road/light conditions, type of collision, pedestrian count, severity, and vehicle count.

  This data will be used for supervised machine learning.

### 2.2 Cleaning the Data
  To get a taste of the data, I did df.head() to see what we have, df.columns to get all the columns, and then df.describe to look for unusual observations. Something I found interesting was that there was an accident that involved 81 people, but not only that, grouping by PERSONCOUNT and VEHCOUNT revealed that it involved only 2 vehicles! Not too sure what happened there, but that is definitely an anomaly. I removed that observation to reduce noise.
  
  There were several variables with a ton of missing values. For the variables that I decided to keep in the end, I decided to impute them with the mode of each variable. This probably introduced slight bias, but in the end the model did slightly better with those imputed values than leaving them blank. I would like to have done some sort of more advanced imputation, but I need to do further research on that.
  
  I removed all observations where PERSONCOUNT = 0 or VEHCOUNT = 0 because it makes no sense to predict the severity of a crash for nobody or no vehicle.
  
  *Removing Outliers*

  After looking at the scatter plot and looking at the mean and standard deviations of PERSONCOUNT and VEHCOUNT, I decided to drop all the observations that were more than 2 standard deviations away from the mean to reduce variability. I calculated what 2 standard deviations above would be for both variables then rounded down and removed all observations that were greater than or equal to that number. I looked at these two variables specifically because I figured that the more people and vehicles involved would probably increase the chance of severity. However, after plotting these two variables together on a scatter plot, it was shown that it did not take 12 vehicles to affect over 20 people. In fact, there were more cases where only 2 vehicles were needed to affect 10 or more people.

  For this, 

  Mean of PERSONCOUNT = 2.44
  STD of PERSONCOUNT = 1.35

  2 STD above = 5.14 ~ 5

  11531 observations where PERSONCOUNT >= 5
  __________________________________________________

  Mean of VEHCOUNT = 1.92
  STD of VEHCOUNT = .63

  2 STD above = 3.18 ~ 3

  16190 observations where VEHCOUNT >= 3


  11531+16190 = 27721 total observations that are considered to be outliers
  
  27721/194673 = .142 ~ 14.2% of total observations dropped from the entire dataset
  
### 2.3 Feature Selection
  
  A quick df.isnull().sum() revealed that there were a significant (defining as >= 40% missing) amount of missing values in the INTKEY, EXCEPTRSNCODE, EXCEPTRSNDESC, INATTENTIONIND, PEDROWNOTGRNT, SDOTCOLNUM, and SPEEDING columns. I decided to automatically drop those columns as there is little to no data to have a chance at being meaningful. I also dropped SEVERITYCODE.1 and SEVERITYDESC because we already have SEVERITYCODE for the labels.
  
  I then inspected the metadata file to get a better idea of what each variable is supposed to represent. I added X, Y, REPORTNO, STATUS, INCDATE, LOCATION, ST_COLCOD, COLDETKEY, OBJECTID to the drop list because I didn't think any of them really added meaningful information. I mean the data is from Seattle, so I didn't think it was singificant to include the coordinates and location. 
  
  This left the variables SEGLANEKEY, CROSSWALKKEY, JUNCTIONTYPE, WEATHER, PEDCOUNT, PEDCYLCOUNT, HITPARKEDCAR, UNDERINFL, ADDRTYPE, COLLISIONTYPE, PERSONCOUNT, VEHCOUNT, ROADACOND, and LIGHTCOND. I got rid of SEGLANEKEY, CROSSWALKKEY, and JUNCTIONTYPE because ADDRTYPE pretty much had all the information summed up and I didn't want a ton of categorical variables. I also got rid of PEDCOUNT and PEDCYLCOUNT since PERSONCOUNT already included that data in it. This helped bring down the number of potential variables. I thought about what to do with WEATHER and ROADCOND since they are essentially the same thing (ex. If the weather is Raining the road is probably Wet); I ended up dropping WEATHER because I felt ROADCOND contributed more information since it had values such as Oil, Sand/Mud/Dirt, and Ice. Finally, I decided to drop HITPARKEDCAR because it didn't seem significant and UNDERINFL because the vast majority of the observations had a N for the UNDERINFL attribute. Doing a quick groupby with SEVERITYCODE and UNDERINFL also revealed that there were more accidents with severity code 1 when UNDERINFL was Y than there were accidents with severity code 2 when UNDERINFL was N.
 
The features I kept are ADDRTYPE, COLLISIONTYPE, ROADCOND, LIGHTCOND, PERSONCOUNT, and VEHCOUNT. After I encoded the categorical variables, I then did a final X.corr() to check that there wasn't an absurd correlation (defining as >= .8) between two variables.

### 2.4 Feature Extraction
  I made a new feature called veh_per_pep, which is supposed to mean vehicles per person. I decided to make this variable because I felt like it would be more informative to have the ratio of vehicles to person and it also helped scale down the values a little. I also tested a decision tree model using this feature and the results were slightly better, so I decided to keep this and drop PERSONCOUNT and VEHCOUNT.
  
  The Final Features used in the model:
  - ADDRTYPE (encoded version)
  - COLLISIONTYPE (encoded version)
  - ROADCOND (encoded version)
  - LIGHTCOND (encoded version)
  - veh_per_pep
  
  In total, the model had 32 features.

## Methodology

### 3.1 Data Exploration 
(all the visualizations will be found in the presentation since I cannot put images in this file)

  I plotted PERSONCOUNT and VEHCOUNT together to find any anomalies. There was an observation that had 81 people involved, but only 2 vehicles were in the accident.  
  
  I ran a correlation matrix on my chosen features (before encoding the categorical variables) to find any signs of multicollinearity between PERSONCOUNT and VEHCOUNT. There wasn't any.
  
  While experimenting with feature extraction, I discovered that there were some observations that had VEHCOUNT = 0 or PERSONCOUNT = 0, which kind of makes no sense, so I removed those observations since it didn't seem like it would be practical to use in this model.
  
  
### 3.2 Machine Learnings

  I implemented KNN, Decision Tree, and Logistic Regression to see which algorithm would give the most accurate results.
  
  The KNN model had a best accuracy of 76.725% at k = 28. Intuitively I thought that KNN would be a good choice under the assumption that crashes that happen under similar circumstances would essentially have the same severity code, but since this performed the worst out of the three models I thought to myself why. My conclusion is that we did not have any data on things like type of cars involved or the age of the people involved. With those variables, I feel like knn would work better because those variables would definitely shed light on how severe a crash could be for someone.
  
  The Decision Tree model had a best accuracy of 77.312% at depth = 3. This model took the shortest time to train and did pretty well. I guess a deicison tree would be appropriate for this dataset since there were only two different severity codes and the model would be learning by following different routes.
  
  The Logistic Regression model had a best accuracy of 77.315% with parameters C = .001, solver = liblinear, and penalty = l1. I think this model makes the most sense, not because it performed the best, but this dataaset only 2 labels: 1 or 2. This is good for logistic regression and logistic regression also gives probabilties about what label each observation could have and using those probabilties is how it makes decisions. 
  
## Results

### 4.1 Evaluation Metrics

  For the 3 models, I calculated 2 different evaluation metrics, Jaccard score and F1-score, for KNN and Decision Tree, and for Logistic Regression I calculated those two and also Log Loss.
  
  The results were: 
  
  KNN: 0.77 Jaccard, 0.73 F1-Score
  
  DecisionTree: 0.77 Jaccard, 0.72 F1-Score 
  
  Logistic Regression: 0.77 Jaccard, 0.72 F1-Score, .46 LogLoss       
  
  All 3 models had the same Jaccard score, but KNN had the highest F1-Score by only .01. Overall, I would choose Logistic Regression to solve this problem.
  
### 4.2 Confusion Matrix and Classification Report Insights
  After plotting the confusion matrices for each model, I found that all 3 models had difficulty predicting a crash with severity code 2 correctly. KNN did the worst at that. All 3 models did really well at predicting accidents with severity code 1 though. Since I don't have a true test set, I cannot train my model on the whole dataset and then run it on the test set to see how well it performs, so these metrics are on the test set derived from the original dataset.
  
  The confusion matrices and classifcation reports will be in the presentation.
  
  
## Discussion

### 5.1 Observations

  Some observations I noted were that it doesn't take a whole lot of vehicles to impact 10+ people since most of the accidents that involved more than 10 people happened when only 2 cars were involved. I also noted that there were some observations that had 0 people or vehicles involved in the accident. Those didn't make much sense at all.
  
### 5.2 Recommendations
  I would like to have imputed the data in a better way that would have led to some sort of improvement rather than just imputing the mode. Doing some sort of distance based imputation would probably have been ideal. If there is any way to get data on the type of vehicle or age of the people involved in the accident then that would probably help out a lot. I would also have liked to seen more than just severity codes 1 and 2. In the metadata, it referenced severity codes 2b and 3. I feel like that would have helped and it would be more insightful for this sort of project.
  
## Conclusion

### 6.1 Concluding Thoughts
  In this project, I aimed to develop a model that would help people see how severe a crash would be if they get in one under certain circumstances. Through feature selection I used ADDRTYPE, COLLISIONTYPE, ROADCOND, LIGHTCOND, PERSONCOUNT, and VEHCOUNT. Then I did feature extraction and made a new variable called veh_per_pep, which is supposed to be the ratio of vehicles to people. The final model used the encoded versions of ADDRTYPE, COLLISIONTYPE, ROADCOND, and LIGHTCOND and the new feature veh_per_pep. The type of model that I decided would be best is a Logistic Regression model. This model can be a starting point to develop a way to inform people about the severity of a crash they could be in. Since it is a Logistic Regression, it can give probabilities about the different types of severity possibilities to be displayed on a screen.
 

  
  
