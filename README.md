# Concrete Compressive Strength Prediction

The quality of concrete is determined by its compressive strength, which is measured using a conventional crushing test on a concrete cylinder. The strength of the concrete is also a vital aspect in achieving the requisite longevity. It will take 28 days to test strength, which is a long period.
I solved this problem using Data science and Machine learning technology, developed a web application which predicts the "Concrete compressive strength" based on the quantities of raw material, given as an input. Sounds like this saves a lot of time and effort right !

`Data source:-` https://www.kaggle.com/elikplim/concrete-compressive-strength-data-set

## Approach:
1. Loading the dataset using Pandas and performed basic checks like the data type of each column and having any missing values.
2. Performed Exploratory data analysis:
    - First viewed the distribution of the target feature, "Concrete compressive strength", which was in Normal distribution with a very little right skewness.
    - Visualized each predictor or independent feature with the target feature and found that there's a direct proportionality between cement and the target feature while there's an inverse proportionality between water and the target feature.
    - To get even more better insights, plotted both Pearson and Spearman correlations, which showed the same results as above.
    - Checked for the presence of outliers in all the columns and found that the column 'age' is having more no. of outliers. Removed outliers using IQR technique, in which I considered both including and excluding the lower and upper limits into two separate dataframes and merged both into a single dataframe. This has increased the data size so that a Machine learning model can be trained efficiently. 
3. Experimenting with various ML algorithms:
    - First, tried with Linear regression models and feature selection using Backward elimination, RFE and the LassoCV approaches. Stored the important features found by each model into "relevant_features_by_models.csv" file into the "results" directory. Performance metrics are calculated for all the three approaches and recorded in the "Performance of algorithms.csv" file in the "results" directory. Even though all the three approaches delivered similar performance, I chose RFE approach, as the test RMSE score is little bit lesser compared to other approaches. Then, performed a residual analysis and the model satisfied all the assumptions of linear regression. But the disadvantage is, model showed slight underfitting.
    - Next, tried with various tree based models, performed hyper parameter tuning using the Randomized SearchCV and found the best hyperparameters for each model. Then, picked the top most features as per the feature importance by an each model, recorded that info into a "relevant_features_by_models.csv" file into the "results" directory. Built models, evaluated on both the training and testing data and recorded the performance metrics in the "Performance of algorithms.csv" file in the "results" directory.
    - Based on the performance metrics of both the linear and the tree based models, XGBoost regressor performed the best, followed by the random forest regressor. Saved these two models into the "models" directory.
4. Deployment:
    Deployed the XGBoost regressor model using Flask, which works in the backend part while for the frontend UI Web page, used HTML5.

At each step in both development and deployment parts, logging operation is performed which are stored in the development_logs.log and deployment_logs.log files respectively. 

So, now we can find the Concrete compressive strength quickly by just passing the quantities of the raw materials as an input to the web application 😊. 


## Web Deployment
Deployed on web using koyeb (PaaS) 
url:- https://eventual-elnore-sanketjagtap-7e11777b.koyeb.app/

## Tools and technologies used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Heroku](https://img.shields.io/badge/heroku-%23430098.svg?style=for-the-badge&logo=heroku&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-%23FF6F00.svg?style=for-the-badge&logo=XGBoost&logoColor=white)
![Random Forest](https://img.shields.io/badge/Random%20Forest-%23FF6F00.svg?style=for-the-badge&logo=Random%20Forest&logoColor=white)
![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white)


## High level design
URL:- 

## Low level design
URL:- 

## Architecture
URL:- 

## Detailed project report
URL:- 

## Wireframe document
URL:- 

## Demo video
URL:- 

  
## References
 
 - [Testing the compressive strength of a concrete in laboratory](https://www.youtube.com/shorts/RMdYjRI-1-c)
 (https://www.youtube.com/watch?v=t4RDdn6rOwU&ab_channel=Anime_Edu-CivilEngineeringVideos) 
 - [Concrete Basics: Essential Ingredients For A Concrete Mixture](https://concretesupplyco.com/concrete-basics/)
 - [Applications of Fly ash](https://www.thespruce.com/fly-ash-applications-844761)
 - [Blast furnace slag cement](https://theconstructor.org/concrete/blast-furnace-slag-cement/23534/)
 - [Applications of Superplasitcizer in concrete making](https://en.wikipedia.org/wiki/Superplasticizer)
 - [Factors that affect strength of concrete](https://gharpedia.com/blog/factors-that-affect-strength-of-concrete/)
 - [Feature selection with sklearn and pandas](https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b)
 - [sklearn's LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)
 - [Post pruning technique in Decision tree algorithm ](https://towardsdatascience.com/3-techniques-to-avoid-overfitting-of-decision-trees-1e7d3d985a09)
 - [Hyper parameter tuning in XGBoost ](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html)
 - [HTML, CSS tutorials ](https://www.w3schools.com/)


## How to Run

To set up and run this project locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/concrete-compressive-strength-prediction.git
   cd concrete-compressive-strength-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```
   python app.py
   ```

5. Open your web browser and navigate to `http://localhost:5000` to access the application.

## Contributing

We welcome contributions to improve this project! Here are some ways you can contribute:

1. Report bugs or suggest features by opening an issue.
2. Improve documentation by submitting pull requests.
3. Add new features or fix bugs by forking the repository and submitting a pull request.

Please ensure that your code adheres to the existing style and that all tests pass before submitting a pull request.

## License

This project is licensed under the MIT License.

MIT License

Copyright (c) 2023 Sanket Jagtap

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

