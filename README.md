# 🍷 Red Wine Quality Predictor
A Machine Learning pipeline to predict the quality of red wine based on physicochemical tests, featuring advanced techniques for imbalanced data and nested cross-validation.

---------
## 📝 About the Project
HI, I'm Susanna! I develop this project for a Machine Learning university exam. The goal is to classify red wine quality (scored from 3 to 8) using various chemical properties such as acidity, residual sugar, and alcohol content. Since extreme quality scores (like 3 or 8) are very rare, the core challenge of the project focuses on handling **severely imbalanced data** while building a robust predictive model that generalizes well to unseen samples.

---------------
## 🛠️ Technologies & Tools
* **Language:** Python 3
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Pipelines, Nested CV, Models, Preprocessing)
* **Imbalanced Data:** `imbalanced-learn` (SMOTE, RandomOverSampler, IMBPipeline)
* **Visualizations:** `matplotlib`
* **Environment:** Deepnote / Jupyter

---------------
## ✨ Features
* **Exploratory Data Analysis (EDA):** In-depth analysis of feature distributions and target class imbalances.
* **Custom ML Pipelines:** Use of `IMBPipeline` and `ColumnTransformer` to seamlessly chain preprocessing, sampling, dimensionality reduction, and classification without data leakage.
* **Imbalanced Class Handling:** Implementation of techniques like SMOTE and RandomOverSampler to address the underrepresentation of excellent and poor wines.
* **Dimensionality Reduction:** Application of PCA (Principal Component Analysis) to retain 95% of the variance while reducing computational cost.
* **Advanced Evaluation:** Use of Nested Cross-Validation for unbiased model selection and Hyperparameter Tuning via `RandomizedSearchCV`.

--------------
## 🚀 Process
1. **Data Exploration:** Analyzed raw data to understand missing values, feature scales, and the skewed distribution of the quality target.
2. **Preprocessing:** Built a custom transformation pipeline to handle scaling, encoding, and imputation.
3. **Model Selection:** Compared different models (including Random Forest and Logistic Regression) using Nested-Cross Validation to avoid overfitting during hyperparameter selection.
4. **Refinement:** Fine-tuned the best performing model (e.g., Logistic Regression with L1 penalty / Random Forest) using hyperparameter search spaces.
5. **Evaluation:** Assessed the final model using robust metrics for imbalanced datasets, such as F1-score and ROC-AUC, rather than just simple accuracy.

-----------
## 🧠 What I Learned
Working on this project was a huge learning experience for me. The biggest hurdle I faced was definitely the class imbalance. Most of the wines in the dataset had an average score (5 or 6), so my models really struggled to identify the rare "excellent" or "poor" wines. It took a lot of trial and error, but learning how to use SMOTE to generate synthetic data finally helped balance things out!

I also realized the hard way why Scikit-learn Pipelines are so important. Before this project, I didn't fully understand how easy it is to accidentally cause data leakage during cross-validation. Setting up a Nested Cross-Validation loop was pretty complex and gave me a few headaches, but it taught me how to evaluate a model properly. Finally, this project taught me to stop relying just on "accuracy": I had to start using metrics like F1-score and ROC-AUC to actually see how my models were doing on those tricky minority classes.

--------------------
## 🔧 Future Improvements
If I had more time to keep working on this project, there are a few things I would love to try:

* Test stronger algorithms: I’d like to experiment with Gradient Boosting models like XGBoost or LightGBM to see if they can beat my current results.

* Feature Engineering: I think the model could improve if I played around with the data more. For example, creating new features by combining existing ones, like the ratio of acidity to alcohol.

* Change the prediction approach: Since wine quality is basically a ranking (from bad to good), treating it as a continuous variable (Regression) instead of strict categories might make more sense.



------------------
## ▶️ How to Run the Project

1. Clone the repository:
   ```bash
   git clone [https://github.com/SusannaMazzocchi/Red_Wine_Quality.git](https://github.com/SusannaMazzocchi/Red_Wine_Quality.git)
   cd Red_Wine_Quality
2. Install the required libraries:
   ```bash
   pip install numpy scipy matplotlib seaborn softpy

3. Open the Notebook:
   ```bash
   jupyter notebook main.ipynb
(Alternatively, you can use Google Colab)
