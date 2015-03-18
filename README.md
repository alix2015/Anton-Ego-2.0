# The Critic
Restaurant reviews analysis and aggregation. The Critic lives
[here](http://the-critic.net)

#### Summary
Restaurant quantitative rating tends to oversimplify the assessment of
restaurants. Even if carried out for different categories
(typically food, service, atmosphere, etc.).
Qualitative rating on the other hand gathers much more detailed information.
Yet how can we process an average of 250 reviews per restaurants for several
thousands of them in the sole city of San Francisco?

 *Water, water everywhere*  
 *Nor any drop to drink*  
 **Coleridge**, The Rime of the Ancient Mariner

In this project, you can find modules to extract the relevant
snippets from a body of reviews according to categories of interest.
You can also perform sentiment analysis on the extracted snippets
with the provided module.


#### Data sources
For this project, I have used data from [Open Table](http://www.opentable.com),
which has detailled rating. I illustrate here the analysis with a small
subset of the obtained data.


#### Installation
You can install this package by cloning the repo.
```
git clone https://github.com/alix2015/the-critic.git
```

Then, you need to start a Mongo server:
```
sudo mongod
```
You can then obtain data (feel free to scrape a subsample of pages):
```
cd back_end/data_acquisition_cleaning/
python scraping_openTable.py
python extract_data.py
```

To run an example, go back to ``back_end`` and run:
```
python main_df.py
```

The front end can be run as follows:
```
cd front_end/app/
python app.py
```

Requirements:
* The data pipeline is built using MongoDB
* Regular Python data science package
  (such as provided with [Anaconda](http://continuum.io/downloads))

* ``dill``
    ```
    pip install dill
    ```
* [TextBlob](http://textblob.readthedocs.org/en/dev/)

* The plotting in the front end is currently done using
  [Plotly](https://plot.ly)


#### Description
Hereafter is a description of the different modules.
The project is organized in two parts:
* back end
* front end

##### Obtaining the data
In the folder [``data_acquisition_cleaning``](https://github.com/alix2015/the-critic/tree/master/back_end/data_acquisition_cleaning),
functions to acquire restaurant reviews from [Open Table](http://www.opentable.com)
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

##### Front end
The front end, which is visible [here](http://the-critic.net), is organized
as a simple site to look up restaurants in the sample set and vizualize
the sentiment distribution and associated snippets for the three main
categories (food, service, ambience) as well as for other categories
calculated as important for the chosen restaurant.
