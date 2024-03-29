# Predicting Netflix Movies rating (user review)

## 1. Introduction

![](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/netflix_picture.jpg)

Netflix, a subscription-based streaming service offers online streaming of films and television series. Starting in 1997 in USA, now it makes the service available in most of the countries in the world.   It is well known for its efficient recommendation engines providing users choice. The engines work behind the scenes and provide user’s choice related contents. The engines use Content-based filtering algorithm, collaborative filtering algorithm or hybrid of both.

User rating(review) plays an important role in recommendation system. The users (called subscribers/viewers/member) rate movies based on various features of a movie: genre, actor, director, title, language, country, duration, production company etc. 
 


## 2. Problem statement

Predictive models will be developed to predict movies users’ rating and the best one will be selected based on the R2-score (co-efficient of determination) i.e how close the actual ratings are to the predicted values.  


## 3. Data:
 
The data was collected from Kaggle and IMDB website. The links are 
* [Kaggle Dataset](https://www.kaggle.com/shivamb/netflix-shows)
* [IMDB Dataset](https://www.imdb.com/interfaces/)


 ## 4. Data Cleaning/wrangling
 [Data wrangling jupyter notebook](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Data_Wrangling.ipynb)

Both datasets list movies and their features as columns. However, based on my requirement, I selected the features those required for the predictive model building. To achieve this, I carried out data wrangling as follows:

**Problem-1:** Kaggle dataset listed both movies and TV shows.  
**Solution:** I separated out the TV shows and only kept the movies for further data processing and model building.   

**Problem-2:** The IMDB dataset has listed titles and original titles both for movies. 
**Solution:** I dropped the ‘title’ column preserving only the ‘original_title’.

**Problem-3:** The movie duration time column had both numbers and unit (minute). 
**Solution:** Unit was deleted and only numerical value was kept.

**Problem-5:** IMDB and Kaggle dataset both contains some columns (date published, metascore, USA gross income, worldwide gross income, budget, reviews from critics) which seemed to be not useful for predictive model building. 
**Solution:** These columns were deleted from respective datasets.

**Problem-6:** After the conversion of the date added column to date, year was extracted. The output obtained was float instead of integer. 
**Solution:** Further investigation was carried out, which showed, it was due to few null values in that column. The null values were filled with 0, and then converted the column to integer.

Finally, all the rows with the null values were removed  from the dataset for EDA (Exploratory Data Analysis). 


## 5. Exploratory Data Analysis (EDA)
[EDA jupyter notebook](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Exploratory_data_analysis.ipynb)
       
Detail investigations into each column of both datasets were carried out and following conclusions were drawn:

•	Both datasets show USA ranked Number One in movie production, while India ranked Second followed by United Kingdom in Third place.
![](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Top_20_countries_in_Movie_production.png)

*	These are the top three movie production companies:

1. Metro-Goldwyn-Mayer 
2. Warner Bros.                 
3. Columbia Pictures   
In USA, India and UK top movie production companies are ‘Metro-Goldwyn-May‘, ‘NH Studioz’ and  ‘The Rank Organisation’ respectively. 

*	Both datasets show most of the movies are of duration (approximately) 90 mins. However, it varies for different countries. In USA and UK movies are of 90-100 minutes, whereas in India movies are of 120-150 minutes.


![](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Overall_movie_duration.png)


![](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/USA_movie_duration.png)

![](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Indian_movie_duration.png)

![](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/UK_movie_duration.png)




## 6. Feature Engineering
[Pre-processing and Taining Data Development jupyter notebook](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Data_Preprocessing_training_data_development.ipynb)

As the IMDB dataset was very large as compared to Kaggle dataset, before feature engineering, I formed two new datasets out of it. Movies, which were found only in IMDB dataset, but not in Kaggle were separated out and used for building the predictive model, whereas movies common to both are used for testing the model. 
Feature engineering was carried out on both training and testing datasets as follows: 
1. Dummie variables for all the categorical columns: genre, language, actors, directors, writers, production company were created. For genre and language, the dummies for all the unique values were carried out, because it listed few varieties (<300),  however for rest of the columns, dummies for top 200 values in each column were obtained.
1. Standard scaling was carried out for numerical columns:  duration_min, votes, reviews from users to keep them in similar range.  As their distribution  follow normal distribution, standard scaling was preferred among all other types of scaling.
1. Few numerical columns had very high outliers, which were not mistakenly entered. However, as it might bias the ML (Machine Learning) model, hence, to avoid this, imputation was carried out, assigning  95th percentile value of the respective column to the outliers.
1. Finally, all the dummies columns were added to the respective main datasets and original categorical columns were dropped.

## 7. Machine Learning (ML) models

To predict the movies rating, different regression models: Simple Linear Regression, Lasso Regression, Ridge Regression, Random Forest Regressor, Gradient Boosting Regressor were developed. 

*	The Simple Linear regression model performance was poor, however it improved significantly when regularization was applied. Among Lasso and Ridge, the later one performed better with R2 score: 0.42. 
* To have better performance, ensemble model Random Forest Regressor was developed. The highest R2 score obtained with this was 0.44.  
* To achieve more higher performance, Gradient Boosting Regressor was developed. The highest R2 score obtained with this was 0.51 and this was the best model among all the models.        
To improve the models’ performance, PCA (Principal Component Analysis) was applied on the data and models were trained again. Though it helped Linear Regression to improve its performance and run time, it couldn’t improve performance of both ensemble models. 

**Best model:** Finally, Gradient Boosting Regressor was saved for deployment and tested on unseen data. 

The two tables below list the details about the ML models. 

![](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/all_models.png)
![](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/gradient_boosting.png)

The Gradient Boosting Regressor model was applied on the unseen movies data and also got R2 score of 0.51.

The model optimization without and with PCA have been discussed in three separate jupyter notebooks Part-1, Part-II Part-III and available in Github in the following links:

[Model Development, Part-I jupyter notebook](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Model_development_Part_I.ipynb)

[Model Development, Part-II jupyter notebook](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Model_development_Part_II.ipynb)

[Model Development, Part-III jupyter notebook](https://github.com/damayantinaik/Springboard_Week_7_Capstone_Project_Netfilx/blob/main/Model_development_with_PCA_Part_III.ipynb)

## 8. Future Recommendations:
In this project, different ML predictive models have been developed to obtain the maximum possible performance with all possible hyperparameter tuning. However, the model performance can further be improved with inclusion of more features like music quality, picture quality, chorography quality, actors’ ranking, users’ age, etc. Hence, I’ll recommend to consider these data for model building in future.

## 9. Acknowledgement
I am grateful to Python developer community for providing many rich, versatile libraries to carry out all types of Data analysis and ML model building. I thank my Springboard mentor Yadunath Gupta for all his thoughtful guidance and constant encouragement to include code in advance pythonic form,  which helped to me to improve myself while working on this project and complete it successfully.  



