# %% [markdown]
# # Chapter 1: The Machine Learning Landscape
#
# ## 1. Chapter Overview
# **Goal:** This chapter provides a high-level overview of Machine Learning (ML),
# defining what it is, why it is useful, and categorizing the various types of ML systems.
# It also covers the typical workflow of an ML project and the primary challenges faced by practitioners.
#
# **Key Concepts:**
# * Definition of Machine Learning.
# * Supervised vs. Unsupervised Learning.
# * Batch vs. Online Learning.
# * Instance-based vs. Model-based Learning.
# * Overfitting and Underfitting.
#
# **Practical Skills:**
# * Loading and preparing data using Pandas.
# * Training a simple Linear Regression model using Scikit-Learn.
# * Comparing Linear Regression with k-Nearest Neighbors.

# %% [markdown]
# ## 2. Theoretical Explanation
#
# ### What is Machine Learning?
# Machine Learning is the science of programming computers to learn from data.
# Instead of explicitly hard-coding rules (e.g., "if email contains 'free', mark as spam"),
# an ML system learns patterns from examples (training data) to make predictions on new data.
#
# ### Types of Machine Learning Systems
# ML systems are generally classified by three criteria:
#
# 1.  **Human Supervision**:
#     * **Supervised Learning:** The training data includes labels (the desired solutions).
#       Examples: Linear Regression, Spam Classification.
#     * **Unsupervised Learning:** The training data is unlabeled. The system tries to learn without a teacher.
#       Examples: Clustering, Dimensionality Reduction.
#     * **Semi-supervised Learning:** A mix of a small amount of labeled data and a lot of unlabeled data.
#     * **Reinforcement Learning:** An agent observes an environment, selects actions, and gets rewards or penalties.
#
# 2.  **Incremental Learning**:
#     * **Batch Learning:** The system is incapable of learning incrementally. It must be trained using all available data offline.
#     * **Online Learning:** The system learns incrementally by feeding it data instances sequentially.
#
# 3.  **Generalization Approach**:
#     * **Instance-based Learning:** The system learns the examples by heart, then generalizes using similarity measures.
#     * **Model-based Learning:** The system builds a model (formula) to make predictions.
#
# ### Main Challenges
# * **Insufficient Quantity of Training Data:** ML algorithms generally need a lot of data.
# * **Nonrepresentative Training Data:** Avoiding sampling bias is crucial.
# * **Poor-Quality Data:** Errors, outliers, and noise hurt performance.
# * **Irrelevant Features:** Success depends on Feature Engineering.
# * **Overfitting:** The model is too complex and memorizes noise.
# * **Underfitting:** The model is too simple to learn the structure.
#
# ### Testing and Validating
# * **Training Set:** Used to train the model.
# * **Test Set:** Used to evaluate the model (estimate generalization error).

# %% [markdown]
# ## 3. Code Reproduction
#
# In this section, we will reproduce the example from the book: **"Does money make people happier?"**
# We will try to predict Life Satisfaction based on GDP per capita.

# %%
# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.neighbors

# %% [markdown]
# ### Data Preparation Helper Function
# The book mentions a `prepare_country_stats` function to merge the OECD Life Satisfaction data
# and the IMF GDP data. We define it here to make the code runnable.

# %%
def prepare_country_stats(oecd_bli, gdp_per_capita):
    # Filter for 'Total' inequality (TOT)
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    
    # Pivot the table to have countries as rows and indicators as columns
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    
    # Rename the GDP column for clarity
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    
    # Merge the two datasets on Country
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    
    # Sort by GDP
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    
    # Return only the relevant columns for our simple model
    return full_country_stats[["GDP per capita", "Life satisfaction"]]

# %% [markdown]
# ### Loading the Data
# We load the data directly from the author's GitHub repository for reproducibility.

# %%
# Load the data
try:
    oecd_bli = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/lifesat/oecd_bli_2015.csv", thousands=',')
    gdp_per_capita = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/lifesat/gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.title("GDP per Capita vs. Life Satisfaction")
plt.show()

# %% [markdown]
# ### Training a Linear Regression Model
# We select a linear model because the data appears to have a linear trend.

# %%
# Select a linear model
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus's GDP per capita
print(f"Prediction for Cyprus (Linear Regression): {model.predict(X_new)[0][0]}")

# %% [markdown]
# ### Training a k-Nearest Neighbors Model
# Alternatively, we can use an instance-based learning algorithm.
# It finds the countries closest to Cyprus in terms of GDP and averages their life satisfaction.

# %%
# Select a k-Nearest Neighbors regression model
clf = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

# Train the model
clf.fit(X, y)

# Make a prediction for Cyprus
print(f"Prediction for Cyprus (k-NN): {clf.predict(X_new)[0][0]}")

# %% [markdown]
# ## 4. Step-by-Step Explanation
#
# ### 1. Data Loading and Preparation
# **Input:** Two CSV files containing OECD life satisfaction data and IMF GDP data.
# **Process:**
# We use `pandas` to read the CSV files. The `prepare_country_stats` function performs an inner join
# on the country names. This ensures we only use countries for which we have both GDP and Life Satisfaction data.
# **Output:** `X` (Feature matrix containing GDP) and `y` (Label vector containing Life Satisfaction).
#
# ### 2. Model Selection
# **Concept:** Model-based Learning vs. Instance-based Learning.
# * **Linear Regression:** We assume a mathematical relationship (Life_Sat = theta_0 + theta_1 * GDP).
#   The `fit()` method calculates the optimal parameters.
# * **k-Nearest Neighbors:** This is instance-based. The model doesn't learn a formula.
#   Instead, it finds the 3 countries with the closest GDP and returns their average life satisfaction.
#
# ### 3. Training (`fit`)
# The `.fit(X, y)` command triggers the learning process.
# * For Linear Regression, it solves a mathematical equation to find the best line.
# * For k-NN, it simply stores the data efficiently.
#
# ### 4. Prediction (`predict`)
# We provide a new instance (`X_new` representing Cyprus). The models output the predicted life satisfaction.

# %% [markdown]
# ## 5. Chapter Summary
#
# * **Machine Learning** is about building systems that learn from data rather than explicit rules.
# * **Workflow:** A typical project involves fetching data, cleaning it, selecting a model, training it, and prediction.
# * **Model Selection:** Choose between model-based (e.g., Linear Regression) or instance-based (e.g., k-NN).
# * **Data Matters:** Quality and quantity of data are often more important than the algorithm.
# * **Evaluation:** Always set aside a **Test Set** to evaluate performance on unseen data.
