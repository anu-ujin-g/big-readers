# Goodreads Book Recommender System
Project for NYU DS-GA 1004 Big Data course
### Authors: 
- Anu-Ujin Gerelt-Od
- Paula Kiatkamolwong
#### Code:
- Initial training of the model on full dataset: train_total.py
- Clean and split the data for training and testing: clean_split.py
- Basic recommender system using ALS: recsys_als.py
- Extension with LightFM for comparison: lightfm_extension.py

### Abstract:
This project explores a basic recommendation system implementation for books, with a comparison to single machine implementations extension. The data set that is used for this project was provided by Goodreads - an online platform for readers to share their opinions on books they’ve read. As we have explicit feedback from the users in the form of a rating, we have used a Collaborative Filtering approach, which creates a recommendation based on the behavior history of the user as well as of users that have similar interests. Additionally, as the data matrix for this type of problem is usually very sparse, we have used an Alternating Least Squares method to process the data set and make predictions.

#### Environment requirements:
- Python
- Apache Spark - PySpark API

