# The Critic
Restaurant reviews analysis and aggregation

#### Summary
Restaurant quantitative rating tends to oversimplify the assessment of
restaurants. Even if carried out for different categories
(typically food, service, atmosphere, etc.).
Qualitative rating on the other hand gathers much more detailed information.
Yet how can we process roughly 250 reviews per restaurants for several thousands
of them in the sole city of San Francisco?

| *Water, water everywhere*  
| *Nor any drop to drink*  
| **Coleridge**, The Rime of the Ancient Mariner

In this project, you can find modules to extract the relevant
snippets from a body of reviews according to categories of interest.
You can also perform sentiment analysis on the extracted snippets
with the provided module.


#### Data sources
For this project, I have used data from [Open Table] (http://www.opentable.com),
which has detailled rating. I illustrate here the analysis with a small
subset of the obtained data.


#### Installation
You can install this package by cloning the repo.
```
git clone https://github.com/alix2015/the-critic.git
```

Requirements:
* The data pipeline is built using MongoDB
* Regular Python data science package
  (such as provided with [Anaconda] (http://continuum.io/downloads))

* ``dill``
    ```
    pip install dill
    ```
* [TextBlob] (http://textblob.readthedocs.org/en/dev/)

* The plotting in the front end is currently done using
  [Plotly] (https://plot.ly) but will soon be upgraded to d3


#### Description
Hereafter is a description of the different modules.
The project is organized in two parts:
* back end
* front end

##### Obtaining the data
In the folder [``data_acquisition_cleaning``] (https://github.com/alix2015/the-critic/tree/master/back_end/data_acquisition_cleaning),
functions to acquire restaurant reviews from [Open Table] (http://www.opentable.com)
are provided, as well as cleaning them. 

##### Extracting data
Extracting the data from MongoDB and transforming it in a usable format.

##### Latent feature extraction and categorization
For this part, I have used TFIDF vectorization, NMF factorization to extract
latent topics in the reviews. For both, I have used scikit-learn implementation.
Latent features are hand-labelled to form categories.
The relevant modules are ``topics`` and ``categories``.

##### Sentiment analysis
I have used TextBlob for sentiment analysis. The analysis is based
on building the dependency tree of sentences. Polarity and subjectivity
are provided by TextBlob. ``sentiment_anlysis`` is the relevant module.
