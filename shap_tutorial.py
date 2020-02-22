# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Introduction

# + [markdown]
'''
Our purpose is to look at how to use and interpret the Shapley values, plots,
    and other information produced by the SHAP package.

To learn more about Shapley values, the SHAP package, and how these are used to
    help us interpret our machine learning models, please refer to these
    resources:


- [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
    > Scott Lundberg, Su-In Lee

- [Consistent feature attribution for tree ensembles](https://arxiv.org/abs/1706.06060)
    > Scott M. Lundberg, Su-In Lee

- [Consistent Individualized Feature Attribution for Tree Ensembles](https://arxiv.org/abs/1802.03888)
    > Scott M. Lundberg, Gabriel G. Erion, Su-In Lee

- [A game theoretic approach to explain the output of any machine learning model.](https://github.com/slundberg/shap)
    >

- [Interpretable Machine Learning:  A Guide for Making Black Box Models Explainable.  5.9 Shapley Values](https://christophm.github.io/interpretable-ml-book/shapley.html)
    > Christoph Molnar, 2019-12-17

- [Interpretable Machine Learning:  A Guide for Making Black Box Models Explainable.  5.10 SHAP (SHapley Additive exPlanations](https://christophm.github.io/interpretable-ml-book/shap.html)
    > Christoph Molnar, 2019-12-17

- [Interpretable Machine Learning with XGBoost](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27?gi=187ef710fdda)
    > Scott Lundberg, Apr 17, 2018

- [Explain Your Model with the SHAP Values](https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d)
    > Dataman, Sep 14, 2019
'''
# -

# # Create a model to interpret

# + [markdown]
'''
First, we'll load some packages.
'''
# -

import pandas as pd
import xgboost
import shap
from sklearn import datasets as ds

# + [markdown]
'''
Next, we'll load some data.  Let's use the California housing data set.
'''

# +
calif_house_data = ds.fetch_california_housing()

print('\n')
print(calif_house_data['data'])
print('\n')
print(calif_house_data['target'])
print('\n')
print(calif_house_data['feature_names'])
print('\n')
print(calif_house_data['DESCR'])
print('\n')
print('Features - # rows, # columns:', calif_house_data['data'].shape)
print('\n')
print('Target variable - # rows:', calif_house_data['target'].shape)

# + [markdown]
'''
The data set has 20640 observations, 8 features, and 1 target variable.  Let's
    put all of that into a data frame.
'''

# +
cah_df = pd.DataFrame(calif_house_data['data'])
cah_df.columns = calif_house_data['feature_names']
cah_df['price'] = calif_house_data['target']

pd.set_option('display.max_columns', 12)
cah_df.head()

# + [markdown]
'''
The target variable *price* is continuous, so we can model it as a regression
    problem.  It and the feature *MedInc* (median income) have apparently
    been log-transformed.  We'll use them as they are, rather than transforming
    them.

Now we can train a model that we can interpret with Shapley values.

We'll create a linear regression model to predict our *price* target variable.

Because we're interested only in how to use Shapley values, we're not going to
    concern ourselves with proper data science practices, like splitting our
    data into training and testing sets or even concerning ourselves much with
    how well our model performs.
'''
# -

x = cah_df.iloc[:, :8]
y = cah_df['price']
model = xgboost.train({'learning_rate': 0.01}, xgboost.DMatrix(x, label=y), 100)
cah_df['predictions'] = model.predict(xgboost.DMatrix(x))


# + [markdown]
'''
Let's take a peek at how our predictions compare to the actual house prices.
'''
# -

cah_df[['price', 'predictions']].head()

# + [markdown]
'''
We can look at the prediction error as a proportion of the target variable.
'''
# -

proportional_error = (cah_df['price'] - cah_df['predictions']).abs() / cah_df['price']
print('\n')
print(proportional_error.mean())
print(proportional_error.median())

# + [markdown]
'''
If we were deeply concerned with how our model is performing, we might
    transform our target variable into actual dollar amounts (which would be
    easier to interpret), summarize our prediction error in dollar terms, plot 
    our errors and analyze for which kinds of houses our model is performing 
    well and poorly, and do other error analyses.  However, while the model is 
    probably overfitting and could be easily improved, we're just going to
    pretend that it's performing well enough and move along to using Shapley
    values to interpret it.
'''
# -

# # Explaining predictions for an individual observation

# + [markdown]
'''
Let's create some SHAP values to explain our model's predictions.
'''
# -

shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x)

# + [markdown]
'''
Notice that the shap values matrix is the same size as our *x* matrix that
    contains all the feature values for each observation (i.e., each row of the
    matrix).
'''
# -

print('\n')
print('Features matrix - # rows, # columns:', x.shape)
print('\n')
print('SHAP values matrix - # rows, # columns:', shap_values.shape)

# + [markdown]
'''
That means that there is one SHAP value for each value in our feature matrix.
    In other words, each observation (row) has a SHAP value for each of its 
    features (columns) that explains that feature's contribution to the model's 
    prediction for that observation.

We can peek at the SHAP values just to see what they look like.
'''
# -

shap_values

# + [markdown]
'''
The SHAP package provides some visualizations to help us understand how much
    each feature contributes to each prediction.  Let's look at a prediction 
    for a single observation (row) in our data set.
The SHAP values explain why a prediction for a single observation is different 
    from the average prediction for all the observations in the data set.
    Here's our model's average prediction for our data set:
'''
# -
explainer.expected_value

# ### Decision plot

# + [markdown]
'''
Here is the SHAP package's *decision plot* for explaining why a single
    observation deviates from the average prediction:
'''
# -

row = 4771
shap.decision_plot(explainer.expected_value, shap_values[row, :], x.iloc[row, :])

# + [markdown]
'''
It's probably easiest to read this plot from the bottom to the top.  At the
    bottom, the squiggly blue line starts at the average prediction for the
    whole data set.  Then as we move up the plot row by row, we're looking at
    each feature's effect on the prediction for our single observation.  If
    the red line moves a lot to the left or right, then the feature for that
    row changes the prediction by a lot.
    
As we move from the bottom to the top of the plot, we notice that *Population*
    and *AveBedrms* (average number of bedrooms) change the prediction very 
    little.  Then *AveRooms* (average number of rooms), *HouseAge* (median
    house age), and *Longitude* all increase the median price of the houses in 
    this census block.
    
The values in parentheses in each row show the value for each feature for the
    houses in this census block.  So, we can see that houses in this block
    have an average of 4 rooms and are 45 years old on average.  Compared to
    all the houses in the data set, these values of 4 rooms and 45 years old
    increase the median price of the houses.
    
As we move further up the plot, we notice that the census block's latitude
    increases the median price, but the average occupancy (*AveOccup*) and
    median income (*MedInc*) decrease the predicted median price, according
    to our model.
    
When we sum up all the SHAP-calculated effects of the features, we see that 
    the model predicts that the median price for houses in this census block
    is 1.37, which is below 1.49, the average predicted median price for all
    houses in all the census blocks in this data set.  that value of 1.37 is
    where the blue squiggly line ends up at the top of the plot.
'''
# -


# ## Force plot

# + [markdown]
'''
The SHAP package provides another type of plot, the *force plot*, to visualize
    the same information as the *decision plot* that we discussed above:
'''
# -

shap.force_plot(explainer.expected_value, shap_values[row, :], x.iloc[row, :])

# + [markdown]
'''
In this *force plot*, the information that we saw in the *decision plot* is
    vertically squashed, or compressed.  The effects of all the features now
    appear on a single row, instead of each feature appearing on its own row.
    This visualization is more compact, but we can still see the same
    information that we saw in the *decision plot*.
    
The average prediction for all houses in all the census blocks (1.49) is labeled 
    as the *base value* here.  The predicted median price for houses in this 
    census block is 1.37 and is labeled as the *output value*.
    
Features that increase the predicted price from the *base value* are in red and 
    are distinguished from each other by arrows pointing upwards.  Features 
    that decrease the predicted price are in blue and have downward-pointing
    arrows.  Features with larger effects on the prediction occupy more space
    in the row of arrows.  The two sets of features point to the *output value*.
    The names of the features and their values are printed below the row of
    arrows.
'''
# -

# # Explaining predictions for the entire data set

# ### Force plot

# + [markdown]
'''
Now that we understand the *force plot* for a single observation, we can look
    at a force plot for many observations.  Our entire data set has 20640 
    observations.  That's too many to plot, so we'll take a random sample of
    those observations for our *force plot*.
'''

# +
n = 200
pd.np.random.seed(16719)
row_idx = pd.np.random.randint(0, x.shape[0], n)

x_sample = x.iloc[row_idx, :]
shap_sample = shap_values[row_idx, :]
# -

shap.force_plot(explainer.expected_value, shap_sample, x_sample, link='logit')

# + [markdown]
'''
In the *force plot* for a single observation, we had a horizontal row of red
    and blue arrows.  For this *force plot* of many observations, the rows of
    red and blue arrows have been rotated so that the arrows for a single
    observation are now vertical.  We can look horizontally across our entire
    sampled data set and easily see approximately how many observations have 
    high predictions or average predictions or low predictions (where the red 
    and blue areas meet). We can also see which features tend to push these
    predictions up or down.
'''
# -
# ### Feature importances summary plot

# + [markdown]
'''
This *force plot* for many observations is terrific for looking at the model's
    predictions with granularity.  But what if we want a simpler summary of
    how important each feature is in making predictions for the entire data 
    set -- something like *feature importance*?
    
The SHAP package provides this as a *summary plot*.  Here it is for our data:
'''
# -

shap.summary_plot(shap_values, x, plot_type='bar')

# + [markdown]
'''
These summaries, or feature importances, are calculated simply by taking the
    absolute values of all the Shapley values and averaging them for each
    feature.  Look closely, and you can see that the calculation below matches
    the plot *summary plot* above.
'''
# -

pd.Series(pd.np.abs(shap_values).mean(axis=0),
          index=cah_df.columns[:x.shape[1]]).sort_values(ascending=False)

# + [markdown]
'''
Just to make sure we understand what's happening here, let's look at the Shapley 
    values again:
'''
# -

print(shap_values)

print('SHAP values matrix - # rows, # columns:', shap_values.shape)

# + [markdown]
'''
Remember, each row is an observation, which represents a census block of houses,
    and we have 20640 census blocks.  Each column is a feature, and there are 8 
    features.
    
All we did in the calculation above was to average all the (absolute values of 
    the) 20640 Shapley values in each column.  That gave us 8 sums, one for each
    feature, and those are our 8 feature importances for this model.  It's that
    simple.
'''

# + [markdown]
'''
Let's pause and consider this for a moment, because this is a really important
    point:  the feature importances for the entire model are calculated 
    directly from their importances for individual observations.  In other
    words, the importances are consistent between the model's global behavior
    and its local behavior.  This consistency is a remarkable and really 
    important characteristic that many model interpretability methods do not 
    offer.  
    
The SHAP package also provides a more granular look at feature importances for
    the entire data set.
'''
# -

shap.summary_plot(shap_values, x)

# + [markdown]
'''
Here the Shapley values every observation are plotted for each feature.
    Additionally, the coloring indicates whether low (or high) values of each
    feature increase (or decrease) the model's predictions.
    
For example, we can see that high median incomes (*MedInc*) increase the
    predictions of house prices (i.e., the Shapley values are greater than zero) 
    while low median incomes decrease those predictions (i.e., the Shapley 
    values are < 0).  Also, the effects of *MedInc* on the model's predictions 
    exhibit a positively skewed distribution:  most values of *MedInc* 
    decrease the model's predictions, while a long tail of high *MedInc*
    values increase the model's predictions.
    
With *AveOccup*, there is a similar effect in the opposite direction.  There
    is a positively skewed distribution so that low values of average occupancy
    increase the model's predictions.  (Note that the univariate distribution of 
    *AveOccup* is negatively skewed, while its effect on the model's 
    predictions are positively skewed.)
'''
# -

# # Explaining predictions for a data set subgroup

# ### Feature importances summary plot

# + [markdown]
'''
 Remember, to calculate the model's global feature importances based on the
     Shapley values, we averaged the (absolute values of the) Shapley values for
     each feature.  That gave us this plot:'''
# -

 pd.Series(pd.np.abs(shap_values).mean(axis=0),
           index=cah_df.columns[:x.shape[1]]).sort_values(ascending=False)

 shap.summary_plot(shap_values, x, plot_type='bar')

# + [markdown]
'''
This additivity, or linearity, property of Shapley values allows us to aggregate
    our explanations across individual observations to explain predictions for
    groups.  In the plot above, we looked at the entire data set, but we can 
    also look at feature importances for subgroups within the data set.
    
 As an example, let's look only at census blocks where houses on average have at
     least 6 rooms.'''
# -

mask = x['AveRooms'] >= 6
shap.summary_plot(shap_values[mask, :], x.loc[mask, :], plot_type='bar')

# + [markdown]
'''
 We can see that for this subgroup, *MedInc* is even more important for 
     generating the model's predictions, compared to *MedInc*'s importance for the
     entire data set.  
 '''
# -

# # Feature interactions

# ### Example 1:  MedInc & AveOccup

# + [markdown]
'''
The SHAP package lets us explore how our features interact with each other to
    produce our model's predictions.  Let's look at an example.
'''
# -

shap.dependence_plot(
    ind=cah_df.columns[0], shap_values=shap_values, features=x, 
    interaction_index=5)

# + [markdown]
'''
We're looking at how median income for the census block (*MedInc*) affects
    the model's predictions.  Ignore the color and *AveOccup* for now.  The
    median income is plotted on the x-axis while its SHAP-measured influence
    on the model's predictions is plotted on the y-axis.  Overall, as median
    income increases, it tends to increase the model's predictions of house
    prices.
    
But notice that that's not the whole story.  If we look at a single value for
    *MedInc*, say where it is equal to 2, notice that the SHAP values have some
    vertical spread.  They range from about -0.8 to -0.3.  If *MedInc* were the
    only predictor influencing its affect on the model's predictions, then all
    the SHAP values where *MedInc* = 2 should be the same.  But they're not;
    that vertical spread tells us that *MedInc* must be interacting with other
    features to produce the model's predictions where *MedInc* = 2.
    
With which features is *MedInc* interacting?  Average occupancy (*AveOccup*) is
    an important one.  Notice that where *MedInc* = 2, *AveOccup* positively
    correlates with *MedInc*'s influence on predictions of house prices.  In 
    other words, when average occupancy is low, *MedInc* tends to predict 
    especially low house prices; when average occupancy is high, *MedInc* 
    predicts house prices that are not as low.  On the plot where *MedInc* = 2, 
    the red dots are higher on the y-axis than the blue dots.  (All the SHAP 
    values where *MedInc* = 2 are negative, so *MedInc* is always decreasing the 
    house price predictions here.)
    
Now look where *MedInc* = 5.  Now the red dots are lower on the y-axis than the
    blue dots.  Now *AveOccup* negatively correlates with *MedInc*'s predictions 
    on house prices.
'''
# -

# ### Example 2:  Latitude & MedInc

# + [markdown]
'''
We can briefly look at another example.  This one looks at how *Latitude*
    interacts with median income (*MedInc*):
'''
# -

shap.dependence_plot(
    ind=cah_df.columns[6], shap_values=shap_values, features=x, 
    interaction_index=0)

# + [markdown]
'''
The influence of *Latitude* on the model's predictions appears to divide into 3
    categories.  The dividing points are a bit north of 34 degrees (the 
    northern part of the Los Angeles metropolitan area) and near 38 degrees (the 
    northern part of the San Francisco metropolitan area).  Generally,
    *Latitude* tends to lower house price predictions as one moves from south
    to north (i.e., from lower to higher *Latitude*).  But there is vertical 
    
But there is vertical dispersion, especially in northern California (at and 
    north of 38 degrees).  *Latitude* interacts with *MedInc*:  *MedInc* 
    negatively correlates with *Latitude*'s influence on house price predictions 
    in southern California (at and south of 34 degrees), positively correlates 
    with it in middle California (between 34 and 38 degrees), and has a less 
    pronounced positive correlation in northern California.  (Remember that the 
    data set is from the 1990 Census, so the model's results may not apply to 
    today's housing markets in California.)
'''
# -

# ### SHAP interaction value matrix

# + [markdown]
'''
We can calculate the SHAP interaction values.
'''
# -

shap_ixn_values = explainer.shap_interaction_values(x)

print(shap_ixn_values[:2, :3, :3])
print('\n')
print('SHAP interaction values - # blocks, # features, # features:', shap_ixn_values.shape)

# + [markdown]
'''
This matrix gives us a SHAP interaction value for every observation (every 
    census block, in this case) and for every combination of features.  This
    way, we can look at how every feature interacts with every other feature
    for every observation.
    
We can also look at the overall feature interactions by aggregating across the
    observations.  (Though I'm not certain that this is the correct calculation.)
'''
# -

ixns_by_features = pd.DataFrame(
    pd.np.abs(shap_ixn_values).sum(axis=0), 
    index=cah_df.columns[:8],
    columns=cah_df.columns[:8])

pd.set_option('precision', 1)
ixns_by_features

# + [markdown]
'''
In the table, we can see that *MedInc* interacts most strongly with *AveOccup*,
    because 795.4 is the highest value in *MedInc*'s column (or row), except for
    *MedInc*'s value with itself (7687.8).
'''
# -

# # Supervised clustering

# + [markdown]
'''
[Lundberg et al. (2019)](arxiv.org/abs/1802.03888) introduced supervised clustering
    based on SHAP values.  This method provides a way of grouping observations 
    in a data set according to how important the features are in producing a
    model's predictions.
    
For our model predicting California house prices, we've already seen that the
    median income of the census block (*MedInc*) is the most important feature
    for generating the model's predictions.  By clustering on the SHAP values
    of the features -- instead of on the features themselves -- *MedInc* will
    automatically be weighted more than the other features in calculating the
    resulting clusters.  Likewise, the other features will be weighted less
    according to their importances.
    
Let's do some supervised clustering.  First, we'll need to load some extra
    packages and functions.
'''
# -

import scipy as sp
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import plotnine as p9


# + [markdown]
'''
To reduce the computational burden, we'll take a sample of our 20640 census 
    blocks.  
'''
# -

n = 400
pd.np.random.seed(19624)
row_idx = pd.np.random.randint(0, x.shape[0], n)

cah_sample = cah_df.iloc[row_idx, :].copy().reset_index(drop=True)
shap_sample = shap_values[row_idx, :]

# + [markdown]
'''
We're going to use a hierarchical clustering method that is conveniently
    available.  However, it's not really important which clustering method we
    choose; you can use any of your favorite clustering methods on SHAP values
    to do supervised clustering.
'''

# +
# returns 1-dimensional order of indices according to hierarchical clustering
# cluster_order_idx = shap.hclust_ordering(shap_sample)

condensed_distance_matrix = pdist(shap_sample)
cluster_linkage_matrix = sp.cluster.hierarchy.complete(condensed_distance_matrix)

# Cophenetic Correlation Coefficient
# sp.cluster.hierarchy.cophenet(cluster_linkage_matrix)

# + [markdown]
'''
For this hierarchical clustering method, let's plot the dendrogram and decide
    how many clusters we want to examine.
'''
# -

plt.figure(figsize=(16, 8))
plt.ylabel('Distance')
plt.xlabel('Index')
sp.cluster.hierarchy.dendrogram(cluster_linkage_matrix)
plt.show()

# + [markdown]
'''
Let's choose a 3-cluster solution, so our maximum distance is going to be about
    1.4.  The code below calculates the clusters, uses T-SNE to reduce the
    dimensionality to 2-dimensions, and then plots those clusters in 2
    dimensions.
'''
# -

max_d = 1.4
tsne = TSNE(n_components=2, verbose=1)
tsne_results = tsne.fit_transform(shap_sample)
clusters = fcluster(cluster_linkage_matrix, max_d, criterion='distance')
tsne_colnames = ['t1', 't2']
clusters_colname = 'clusters'
tsne_results_df = pd.DataFrame(tsne_results, columns=tsne_colnames)
tsne_results_df[clusters_colname] = clusters

plot_title = 'SHAP-Based Clusters in T-SNE SHAP Space'
x_axis_label = 'T-SNE Component 1'
y_axis_label = 'T-SNE Component 2'
xlim = [tsne_results_df.iloc[:, 0].min(), tsne_results_df.iloc[:, 0].max()]
ylim = [tsne_results_df.iloc[:, 1].min(), tsne_results_df.iloc[:, 1].max()]

plot = (p9.ggplot(tsne_results_df,
                    p9.aes(y=tsne_results_df.columns[1], 
                           x=tsne_results_df.columns[0],
                           group=clusters_colname,
                           color=clusters_colname
                           ))
        + p9.geom_point(size=2)
        + p9.geom_rug()
        + p9.stat_ellipse()
        + p9.xlim(xlim[0], xlim[1])
        + p9.ylim(ylim[0], ylim[1])
        #+ p9.scale_color_gradient(low='blue', high='yellow')
        #+ p9.scale_color_manual(values=colors)
        + p9.theme_light(base_size=18)
        + p9.ggtitle(plot_title)
        + p9.labs(y=y_axis_label,
                  x=x_axis_label)
        )

plot_filename = 'shap_clusters.png'
plot.save(plot_filename, width=10, height=10)
from IPython.display import Image
Image(filename=plot_filename)

# + [markdown]
'''
The plot above shows our 3 clusters in different colors and outlined by 
    ellipses.  To understand these clusters, we need to look at some statistics
    and/or plots that characterize them.  Let's take a look at the means for
    our feature values, the feature SHAP values, the house prices, and the
    predicted house prices by cluster (though remember that our clusters are
    based specifically on the feature SHAP values and the predicted house 
    prices).
'''
# -

shap_sample_df_colnames = [e + '_shap' for e in cah_df.columns[:shap_sample.shape[1]]]
shap_sample_df = pd.DataFrame(shap_sample, columns=shap_sample_df_colnames)
cah_shap_sample = pd.concat([cah_sample, shap_sample_df], axis=1)
cah_shap_sample[clusters_colname] = clusters

pd.set_option('display.max_columns', 20)
cah_shap_sample.groupby(clusters_colname).mean()

# + [markdown]
'''
The most obvious difference among the clusters is that Cluster #1 has the
    highest house prices, the highest predicted house prices, and the highest
    median income (*MedInc*), which is also the most important feature in
    generating Cluster #1's high predicted house prices, according to the SHAP
    values for *MedInc* (*MedInc_shap*).  In short, Cluster #1 evidently
    includes the most economically prosperous neighborhoods.
    
By contrast, Cluster #2 has the lowest house prices, the lowest predicted house 
    prices, and the lowest median income (*MedInc*), which is the most important
    feature in lowering the model's predictions of Cluster #2's house prices.
    
Cluster #3 falls between Clusters #1 and #2 on all these statistics.  Notice
    that our T-SNE plot of the clusters capture these relationships, with
    Cluster #3 data points plotted mostly between data points for Clusters #1 
    and #2.
    
If we wanted to examine our data set and our model's predictions in more 
    detail, we might divide our observations into more than 3 clusters and look
    at boxplots, geography, and other ways of characterizing our clusters.  But
    the primary purpose here is only to briefly demonstrate how supervised 
    clustering can be useful in interpreting a model and understanding the 
    underlying data.  [Lundberg et al. (2019)](arxiv.org/abs/1802.03888) provide 
    further details about supervised clustering.
'''
# -

# # Conclusion

# + [markdown]
'''
This concludes our exploration of how to use Shapley values and the SHAP 
    package.  The resources cited at the top of this page can provide further 
    information about Shapley values and SHAP.  Additionally, Christoph 
    Molnar's book and Tim Miller's paper can provide further insight into the
    challenges and promise of machine learning interpretability:
    
- [Interpretable Machine Learning:  A Guide for Making Black Box Models Explainable.](https://christophm.github.io/interpretable-ml-book/)
    > Christoph Molnar, 2019-12-17

- [Explanation in Artificial Intelligence: Insights from the Social Sciences](https://arxiv.org/abs/1706.07269)
    > Tim Miller
'''
