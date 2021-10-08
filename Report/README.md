# Predicting Netflix Movies rating

## 1. Introduction

![](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Report/Netflix_image.jpg)

Netflix, a subscription-based streaming service offers online streaming of films and television series. Starting in 1997 in USA,  now it makes the service available in most of the parts  of the world.   It is well known for its efficient recommendation engines providing users choice. The engines work behind the scenes and provide user’s choice related contents. The engines use Content-based filtering algorithm, collaborative filtering algorithm or both.

User rating(review) plays an important role in recommendation system. The users (called subscribers/viewers/member) rate movies based on various features of  a movie like genre, actor, director, title, language, country, duration, production company etc. History of users choice on different types of movies also helps to predict the rating of the unseen movie. 


## 2. Problem statement

Different predictive models will be developed to predict movies users’ rating and the best one will be selected based on the R2-score (co-efficient of determination) i.e how close the actual ratings are close to the predicted values. 


## 3. Data:
 
The data was collected from Kaggle and IMDB website. The links are 
* [Kaggle Dataset](https://www.kaggle.com/shivamb/netflix-shows)
* [IMDB Dataset](https://www.imdb.com/interfaces/)


 ## 4. Data Cleaning/wrangling
 [Data wrangling jupyter notebook](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Report/Capstone_Project_Netflix_Data_Wrangling_submission4_Report.ipynb)

Both datasets lists movies along with other features as columns. However, based on my requirement, I selected features those required for the predictive model building. To achieve this, I did data wrangling and the following were carried out: 
  
**Problem-1:** Kaggle dataset listed both movies and TV shows. 
**Solution:** I separated out the TV shows and only kept the movies for further data processing and model building.  

**Problem-2:** The IMDB dataset has listed titles and original titles both for movies. 
**Solution:** I dropped the ‘title’ column preserving only the ‘original_title’.

**Problem-3:** The duration time was including both numbers and unit(minute). 
**Solution:** Minute was deleted and only numerical value was kept.

**Problem-5:** IMDB and Kaggle dataset both contains some columns (date_published, metascore, usa_gross_income, worlwide_gross_income, budget, reviews_from_critics) which seems to be not useful for predictive model building. 
**Solution:** These columns were deleted from respective datasets.

**Problem-6:** After the conversion of the date_added column to date, the year was extracted. The output obtained was float instead of integer. 
**Solution:** To find out the reason, further investigation was carried out which shows that this is happening due to the presence of the null values in time stamp. To handle this, the null values were  filled with 0, then converted those to integer.

Finally rows with Null values are removed to prepare the dataset for EDA( Exploratory Data Analysis). 


## 5. Exploratory Data Analysis (EDA)
[EDA jupyter notebook](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Report/Netflix_EDA_submission2_for_report.ipynb)
       
Detail investigations into each column of both datasets were carried out and following conclusions were drawn:
•	Both datasets show USA ranked Number One in movie production, while India ranked Second followed by United Kingdom in Third place.
![](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Report/Top_20_countries_in_Movie_production.png)

•	These are three movie production companies:
1.Metro-Goldwyn-Mayer 
2. Warner Bros.                 
3. Columbia Pictures             
In USA, India and UK top movie production companies are ‘Metro-Goldwyn-May‘, ‘NH Studioz’ and  ‘The Rank Organisation’ respectively. 
•	Both datasets show most of the movies are of length (approximately) 100 mins. However, it varies for different countries. In USA and UK movie length is ~100 minutes, whereas in India movies are of 150 minutes.


![](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Report/Overall_movie_duration.png)


![](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Report/USA_movie_duration.png)

![](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Report/Indian_movie_duration.png)

![](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Report/UK_movie_duration.png)




## 6. Feature Engineering
[Pre-processing and Taining Data Development jupyter notebook](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Report/Netflix_data_Pre_processing_training_data_development_submission2_Report.ipynb)

As the IMDB dataset was very large as compared to Kaggle dataset, before feature engineering, I formed two new datasets out of it. Movies, which were found only in IMDB dataset, but not in Kaggle were separated out and used for building the predictive model, whereas movies common to both are used for testing the model. 
Feature engineering was carried out on both training and testing datasets as follows: 
1. Dummie variables for all the categorical columns: genre, language, actors, directors, writers, production company were created. For genre and language, the dummies for all the unique values were carried out, because it listed few varieties (<300),  however for rest of the columns, dummies for top 200 values in each column were obtained.
1. Standard scaling was carried out for numerical columns:  duration_min, votes, reviews from users to keep them in similar range.  As their distribution  follow normal distribution, standard scaling was preferred among all other types of scaling.
1. Few numerical columns had very high outliers, which were not mistakenly entered. However, as it might bias the ML (Machine Learning) model, hence, to avoid this, imputation was carried out, assigning  95th percentile value of the respective column to the outliers.
1. Finally, all the dummies columns were added to the respective main datasets and original categorical columns were dropped.

## 7. Machine Learning (ML) models

To predict the movies rating, different regression models: Simple Linear Regression, Lasso Regression, Ridge Regression, Random Forest Regressor, Gradient Boosting Regressor were developed. 
•	The Simple Linear regression model performance was poor, however it improved significantly when regularization was applied. Among Lasso and Ridge, the later one  performed better with r2_score (test set): 0.42. 
•	To have better performance, ensemble model Random Forest Regressor was developed. The highest r2_score obtained with this was 0.44. 
•	To obtain more higher model performance, Gradient Boosting Regressor was developed. The highest r2_score obtained with this was 0.507 and this was the best model among all the models.    
To improve the models’ performance, PCA (Principal Component Analysis) was applied on the data and models were trained again. However, though it helped Logistic Regression to improve its performance and run time, it couldn’t improve performance of both ensemble models. 

**Best model:** Gradient Boosting Regressor was saved for deployment and tested on unseen data. 


![]()


Below is a table listing the best features, hyperparameters, and metrics of the best model:
model	features	hyperparameters	R2_score
Gradient boosting (best model)	700	Learning rate : 0.1
n_estimators: 700
max_depth: 7	0.507(train)
0.47(test)

The final Gradient Boostimng model with r2_score : 0.50 was applied on the unseen movies to determine their ratings. The model performance was very close to train dataset with r2_score: 0.47.

The model optimization without and with PCA have been discussed in three separate jupyter notebooks as Model Development Part-1, Part-II Part-III and available in Github in following links:

[Model Development, Part-I jupyter notebook](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Report/Netflix_data_model_development_final_Part_I_Report.ipynb)

[Model Development, Part-II jupyter notebook](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Report/Netflix_data_model_development_final_Part_II_Report.ipynb)

[Model Development, Part-III jupyter notebook](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Report/Netflix_data_model_development_PCAall_final_Part_III_Report.ipynb)

## 8. Future Recommendations:
In this project, all the possible ML predictive models have been applied to obtain the best model performance with maximum possible hyperparameter tuning. However, I believe, the model performance can be further improved with addition of more movie features like 'music quality, picture quality, chorography quality, actor ranking, user age,  etc. Hence, in future,  datasets on these should also be included for analysis and ML model building. 

## 9. Acknowledgement
I am grateful to Python developer community for providing many rich, versatile libraries to carry out all types of Data analysis and ML model building. I thank my Springboard mentor Yadunath Gupta for all his thoughtful guidance and constant encouragement to include code in its advance pythonic form which helped to me to improve myself while working in this project and complete successfully.  



