# Technology Sector Salary Analysis

## Introduction
<p style="font-size:18px;">To give the sutdents in this consortium an idea of what is possible for career possiblities after
completion. Showing the different job titles, experience levels and what possible salary ranges could be earned. Showing what 
job titles are trending from the dates 2020-2024.</p>

<p style="font-size:18px;">We also will be looking at various multi-class classification models and determining which is the most accurate model.


## Table of Contents
<ol style="font-size:18px; font-style:italic;">
  <li><a href="#Introduction">Introduction</a></li>
  <li><a href="#Prerequesistes">Prerequesistes</a></li>
  <li><a href="#Data">Data</a></li>
  <li><a href="#Methodology">Methodology</a></li>
  <li><a href="#Optimization">Optimization</a></li>
  <li><a href="#Results">Results</a></li>
  <li><a href="#Conclusion">Conclusion</a></li>

  <li><a href="#Future-Work">Future Work</a></li>
  <li><a href="#Reference">Reference</a></li>
  <li><a href="#Acknowledgements">Acknowledgements</a></li>
</ol>

## Prerequisites ( software, libraries and tools)
- 'Pandas' For data manipulation and analysis
- 'Gradio' For building machine learning and data science demos
- 'Scikit-Learn' for machine learning algorithms and tools
- 'numpy' For numerical operations and calculations
- 'pathlib' For file path management
- 'matplotlib' For static plotting of data
- 'seaborn' For sstatistical data visualization
- 'scipy.stats' For statistical analysis
- 'imbalanced-learn' For handling imbalance datasets
- 'pydotplus' For visulaization of decision trees

## Data
<p style="font-size:18px;">The Data Science Salaries dataset from Kaggle provides comprehensive insight into the compenssation trends in the data science
 fields across 75 countries. This dataset is our resource for analyzing and predicting salary based on various factors such as job titles and experience levels.
 The salaries are recorded in multiple currencies, buty converted to USD for consistency in analysis. 

## Methodology
**Classification**
- Create a histogram for numberical columns to determine data distribution.
- Use the 'salary_in_usd' column to show a normal distribution for a linear regression model.
- Create 4 bins based on salary distribution
- Adjust bins for more balanced employee count.
- Stored non-collinear features (job_title, experience_level, employment_type, work_models, employee_residence, company_location, company_size) in X
- Store salary categories in y.
- split data into 75% training and 25% testing sets.
- Train 5 classification models: K Nearest Neighbors(KNN), Support Vector, Decision Tree, Random Forest and AdaBoost Classifers.

**Linear Regression**
- Remove outlier salaries above $650,000 to prevent extreme values from skewing the model.
- apply a natural logarithm transformation to the 'salary_in-usd' to normalize the salary distribution, making it more suitable for linear regression.
- Select relevant features and remove unnecessary colums and transform 'salary_in_usd'.
- Use OneHotEncoder to convert categoriaal features into numerical format while, ignore unknown categories.
- Apply StandardScaler to standardize numeric features to improve model performance.

**Unsupervised Learning**
- Load the CSV file containing salary data into a DataFrame
- Remove columns 'salary', 'salary_currency, and 'work_year"
- Compute quartiles (Q1, Q3) and interquartile rage(IQR) of 'salary_in_usd.
- Identify outliers using the lower and upper bounds base on IQR
- Exclude data points outside the calculated bounds.
- Encode categorical columns with high cardinality base on  their frequency disctribution.
- Apply one-hot encoding to categorical columns with low cardinality.
- Scale the 'saly_in_usd column using StandardScaler.
- Reduce the dimensionality of the data to three principal components using PCA
- Determine the optimal number of clusters using the Elbow method
- Fit the KMeans model using the optiman number of clusters and predict the customer labels.
- Fit the Agglomerative Clustering model and predict the cluster labels.
- Fit the Birch model and predict the cluster labels.

## Optimization - need to show interative changes
**Classification**
- Use Random Oversampling to duplicate minority class examples.
- Use Random Undersampling to remove majority class examples.
- Fit and resample the X-Train and y-Train, then retrain and retest models.

**Linear Regression**
- Choose the Ridge Regression, a variation of linear regression that includes a penalty term to prevent overfitting and manage multi ollinearity in datasets with many festures.
- Use the r2Score and the Mean Squared Error (MSE) to test the models Performance and Accuracy.

**Unsupervised Learning**
- KMeans: Use the Elbow method to determine the optial number of clusters by plotting inertia values for differnt values of k.
- Agglomerative Clustering: Number of clusters is set to 3.
- Birch: Number of clusters is set to 3.

## Results
**Classification**
- All models performed worse with oversampling and undersamplint
- AdaBoost perfromed the best with 61.39% accuracy. It correcly identified 83% of actual average salaires but misclassified other categories as average.
- Precision of 0.54 and recall of 0.80 for Class 1, recall of 0.64 for Class 2, and recall of 0.45 for Class 3.

**Linear Regression**
- Test r2 Score: 0.5211
- Test MSE: 0.1434

**Unsupervised Learning**
- Plot the inertia values for different values of k to visually identify the optimal value for k.
- Add cluster labels to the DataFrame for KMeans, Agglomerative Clustering, and Birch models.
- Plot the clusters for each model using scatter plots.
- Retrieve the explained variance ratio to determine how much information can be attributed to each principal component.
- View the component weights for each principal component to undertand the contributions of each feature.

*Additional Insights*
- Calculate mean, median, and count of 'salary_in_usd' grouped by "work_models'.
- Calculate mean, median, and count of 'salary_in_usd' grouped by 'expericane_level'.
- Calculate correlations between 'salary_in_usd' and other features.
- Save the precessed DataFrame to a CSV file for futher analysis or sharing.

## Conclusion
**Classification**
- Signigicantly impacted model performance, causing bias toward dominant classes
- The current models, particulary AdaBoos, were not reliable due to poor generalization and class imbalance.

**Linear Regression**
- The Ridge Regression model effectively predicts salary values based on  job related inputs

**Unsupervised Learning**
- Optimal Clusters: The Elbow method suggest the optimal number of clusters for KMeans.
- Cluster Patterns: Visualizations reveal distinct clusters for job titles and standardized salary data.
- Model Comparison: Different Clustering algoriths (KMeans, Agglomerative, Birch) provide varying insights into the data structure.
- PCA Analysis: PCA helps in reducing the dimensionality of the data while retaining significant informations.
- Data Preparation: The cleaned, encoded and dimensionally-reduced dataset is effectively used to identify meaningfule patterns through clustering.

**Low classification accuracy and R Score**
- The complexity and high variablity of the data contributed to the lower scores.
- Even though extreme outliers were removed, there may be smaller outliers that can affect the performance.
- There was also class imbalance and even using the oversampling, undersampling techniques the model would still not achieve the desired accuracy.
- An r2 score of 0.5211 suggests that the model explains about 52% of the varaiance in the salary data. The moderate level might be due to missing factors.

## Gradio Interface
- User selects relevant job detail.
- Input data is encoded using OneHotEncoder during model training.
- Predict the exact salary in USD.
- Display Predicted salary as an output.

## Future Work
<p style="font-size:16px;">To predict overall compensation more accurately, we would like to include more factors that can influnce salary levels. Some additional data points
that would be useful:
- Years of experience
- Education level
- Industry
- Skills and Certifications
- Job Function and Responsibilities
- Perfromance Bonuses and Incentives
- Stock options and Equity
- Benefits and Perks

Consider alterantive resampling techniques, feature engineering and model tuning. testing with different algorithms and ensembe method could also help address class imbalance and improve overall performance.

## References
- [Data Science Salaries 2024 Dataset on Kaggle](https://www.kaggle.com/datasets/some-dataset-link)

## Acknowledgements
<h1 style="color:#89CFF0; font-size:36px; text-align:center;">The Fantastic Four</h1>

<ul style="font-size:20px;">
  <li><strong>Daniel Liu</strong></li>
  <li><strong>Ryan Brown</strong></li>
  <li><strong>Tony Lockhart</strong></li>
  <li><strong>Cathy Schassberger</strong></li>
