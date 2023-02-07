# SHAP tutorial

Our purpose is to look at how to use and interpret the Shapley values, plots, and other information produced by the SHAP package.


# Note

Currently, some plots in the notebook 'shap_tutorial.ipynb' may not render properly on Github's website.  As an alternative, one can download the notebook and view it locally or download and view its HTML version 'shap_tutorial.html'.


# Background from other resources

To learn more about Shapley values, the SHAP package, and how these are used to help us interpret our machine learning models, please refer to these resources:


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


Christoph Molnar's book and Tim Miller's paper can provide further insight into the challenges and promise of machine learning interpretability:
    
- [Interpretable Machine Learning:  A Guide for Making Black Box Models Explainable.](https://christophm.github.io/interpretable-ml-book/)
    > Christoph Molnar, 2019-12-17

- [Explanation in Artificial Intelligence: Insights from the Social Sciences](https://arxiv.org/abs/1706.07269)
    > Tim Miller


For my own blog post describing how machine learning interpretability can be used in healthcare, please see:
    
- ORIGINAL LINK BROKEN:  <s>[Interpretability and the promise of healthcare AI](https://www.geneia.com/blog/2020/january/interpretability-and-the-promise-of-healthcare-ai)</s>
- UPDATED LINK: [Interpretability and the promise of healthcare AI](https://web.archive.org/web/20210418122745/https://www.geneia.com/blog/2020/january/interpretability-and-the-promise-of-healthcare-ai)
    > Andrew Fairless, Ph.D., Principal Data Scientist, January 23, 2020 

