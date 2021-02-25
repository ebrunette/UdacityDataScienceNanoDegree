# AirBnB Dataset Analysis 

# Required Libraries for running the project
1. pandas
2. seaborn
3. matplotlib
4. numpy
5. sklearn
6. datetime
7. math

# Motivation for the project
Aside from the obvious motivation of completing the nano degree, the analysis is centered around a main theme. How can those that have AirBnB locations figure out how to buy their new rental location or improve the nightly price of their current rentals. This analysis also explains some key factors that decrease the price of their rentals. 

# Files in the repo
### AirB&BEDA.ipynb
* This notebook expresses analysis around the Dataset provided by AirBnB through the Udacity Blog post project.
* Accompaning article can be found [here](https://elibrunette.medium.com/factors-to-increase-rental-prices-for-airbnb-6a4cbb928e0d) 
### data/archive/calendar.csv
* This file contains the calendar availability and daily rates for each of the rentals. 
### data/archive/listings.csv
* This file contains the listings for the properties, along with various other features that were analyzed in the jupyter notebook. 
* This file also contains the review scores, prices, and various other features pertaining to the listings. 
### data/archive/reviews.csv
* This file contains the raw text reviews for each of the listings. 
### screenshots 
* This folder contains visualizations from the analysis and screenshots that are pulled from the run analysis jupyter notebook. 

# Results of the analysis
### 1. Does having more pictures of the house correspond to higher overall reviews or prices for the location?
1. After some review it appears there is a good coorelation between including pictures and ratings.
2. This could be just reflecting the overall distribution of reviews for the data though without showing any clear signs of one being better than the other. This is shown in the heatmap from the step 1 analysis. <br>
![Question1Results](https://github.com/ebrunette/UdacityDataScienceNanoDegree/blob/master/ProjectOne/screenshots/HeatmapOfPictureDistribution.PNG)

### 2. Is square feet correlated to price?
1. Since there is only 97 rows with square feet not null, this questions will be excluded from investigation
2. However, from the data present, there isn't any notable correlation between price and square feet <br>
![Question2Results](https://github.com/ebrunette/UdacityDataScienceNanoDegree/blob/master/ProjectOne/screenshots/SquareFeetVisual.PNG)

### 3. What feature correlates to higher prices?
1. Based on the coefficients for the linear model, it looks like the three higblob/attributes to determine price is bathrooms, bedrooms and beds.

![Question3Results](https://github.com/ebrunette/UdacityDataScienceNanoDegree/blob/master/ProjectOne/screenshots/FeatureBarChart.PNG)
### 4. What feature correlate to higher overall review ratings?
1. From visual 3. and analysis the main features that contibute to price is thblob/ber of bedrooms, bathrooms, bed, and review scores for the location. 

### 5. What days are the most popular? Specifically Weekdays or Weekends?
1. The data suggests that there is a correlation to the weekends having higherblob/ rate for most of the stats by $5.
![Question5aResults](https://github.com/ebrunette/UdacityDataScienceNanoDegreeblob//master/ProjectOne/screenshots/WeekdayPriceHistogram.PNG)
![Question5bResults](https://github.com/ebrunette/UdacityDataScienceNanoDegreeblob//master/ProjectOne/screenshots/WeekendPriceHistogram.PNG)

### 6. Does higher review count correspond to higher prices on the location?
1. Based on the review from 3, it does not appear so. Actually the review scores negatively attribute to price. I believe this is because of the number of highly rated properties and listings in the dataset.

### 7. Does not having a picture correlate to not having a review?
1. It does not. There is a good portion of na reviews that have pictures. I again believe that this has something to do with the overall distribution of the data for the dataset.

# Acknowledgements
* Thank you to Udacity for helping me refine my analysis skills and providing the platform for obtaining the data analyzed. 
* Thank you to AirBnB for providing the data that I am using for analysis in this repo. 