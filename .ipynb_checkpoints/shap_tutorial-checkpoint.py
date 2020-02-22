# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
## Set up model to interpret

# %% [markdown]
'''
First, we'll load some packages.
'''

# %%
import os
import pandas as pd
import xgboost
import shap
from sklearn import datasets as ds

# %% [markdown]
'''
Next, we'll load some data.  Let's use the California housing data set.
'''

# %%
calif_house_data = ds.fetch_california_housing()

print(calif_house_data)
print('\n')
print(calif_house_data['data'].shape)
print('\n')
print(calif_house_data['target'].shape)

# %% [markdown]
'''
The data set has 20640 observations, 8 features, and 1 target variable.  Let's
    put all of that into a data frame.
'''

# %%
cah_df = pd.DataFrame(calif_house_data['data'])
cah_df.columns = calif_house_data['feature_names']
cah_df['price'] = calif_house_data['target']

pd.set_option('display.max_columns', 12)
cah_df.head()

# %% [markdown]
'''
The target variable 'price' is continuous, so we can model it as a regression
    problem.  However, let's also do some classification, so let's create a
    binary varaible of this target variable.  We'll split 'price' at its 
    median.
'''

# %%
print(cah_df['price'].median())

cah_df['price_binary'] = cah_df['price'] > cah_df['price'].median()

cah_df['price_binary'].value_counts()

# %% [markdown]
'''
Now we can train a model that we can interpret with Shapley values.

We'll create a logistic regression model to predict our dichotomized 'price'
    target variable.

Because we're interested only in how to use Shapley values, we're not going to
    concern ourselves with proper data science practices, like splitting our
    data into training and testing sets.
'''

# %%
x = cah_df.iloc[:, :8]
y = cah_df['price_binary']
model = xgboost.train({'learning_rate': 0.01}, xgboost.DMatrix(x, label=y), 100)
cah_df['prediction_prob'] = model.predict(xgboost.DMatrix(x))
cah_df['predictions'] = cah_df['prediction_prob'] >= 0.50

model_correct_n = (cah_df['price_binary'] == cah_df['predictions']).sum()

print(model_correct_n)
print(cah_df.shape[0])
print(model_correct_n / cah_df.shape[0])

# %% [markdown]
'''
Our model predicts the target variable with 87% accuracy on the same data that
    we trained it on.  While the model is probably overfitting, that performance 
    is plenty good enough for our purpose of interpreting the model.
'''

# %% [markdown]
# # section 2 title

# %% [markdown]
'''
Let's create some SHAP values to explain our model's predictions.
'''

# %%
shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x)

# %% [markdown]
'''
Notice that the shap values matrix is the same size as our 'x' matrix that
    contains all the feature values for each observation (i.e., each row of the
    matrix).
'''

# %%
print(x.shape)
print('\n')
print(shap_values.shape)

# %% [markdown]
'''
That means that there is one SHAP value for each value in our feature matrix.
    In other words, each observation has a SHAP value for each of its features
    that explains that feature's contribution to the model's prediction for
    that observation.
'''

# %% [markdown]
'''
We can peek at the SHAP values just to see what they look like.
'''

# %%
shap_values


# %%
row = 0
shap.force_plot(
    explainer.expected_value, 
    shap_values.iloc[row, :], 
    x.iloc[row, :], link='logit')






