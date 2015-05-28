GitAnalysis
===========

Code for Git Analytics

Copyright Doug Williams, 2015


Development notes and summary results can be found in [_README Notebook](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/_README.ipynb)

Sample analysis data can be found at [williamsdoug/GitAnalyticsDatasets](https://github.com/williamsdoug/GitAnalyticsDatasets)
- [Various Dataset Sizes](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Dataset_Sizes.ipynb)
- See: IPython notebook [OpenStack_Sample_Data.ipynb](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/OpenStack_Sample_Data.ipynb) for examples of each record format.

Analysis Notebooks for Various OpenStack Projects:
- [Cinder](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Analysis_of_Cinder.ipynb)
- [Glance](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Analysis_of_Glance.ipynb)
- [Heat](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Analysis_of_Heat.ipynb)
- [Nova](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Analysis_of_Nova.ipynb)
- [Swift](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Analysis_of_Swift.ipynb)


Performance of individual solvers:

- [AdaBoost](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Curves_AdaBoost.ipynb)
- [DecisionTree](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Curves_DecisionTree.ipynb)
- [ExtraTrees](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Curves_ExtraTree.ipynb)
- [GradientTreeBoosting](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Curves_GradientTreeBoosting.ipynb)
- [Logistic Regression](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Curves_LogisticRegression.ipynb)
- [Naive Bayes - Gaussian](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Curves_NaiveBayes.ipynb)
- [Random Forest](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Curves_RandomForest.ipynb)
- [Stochastic Gradient Descent](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Curves_SGD.ipynb)
- [Support Vector Machines](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Curves_SVM.ipynb)
- [Neural Networks using Theano](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Curves_Theano_NN.ipynb)


Others, may be a bit rough:
- [Composite learner using boosting](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Composite_Learner.ipynb)
- Neural Networks using Multi-Layer Perceptron:[First Attempt](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/MLP_Round_1.ipynb) and [Second Attempt](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/MLP_Round_2.ipynb) and [Various topologies using Theano](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Curves_Theano_NN-NetworkSize.ipynb)

Python source code located in: ./dev

Configuration file located at: ./dev/git_analysis_config.py

Dataset build script: [Build_All_Datasets.ipynb](http://nbviewer.ipython.org/github/williamsdoug/GitAnalysis/blob/master/notebooks/Build_All_Datasets.ipynb)


Changes
=======

3/12/2015: Major update
5/28/2105:  Clean-up notebooks, new python aware diff routine, language-specific features, Theano-based NN