# The Critic
Restaurant reviews analysis and aggregation

#### Summary
Restaurant quantitative rating is carried out according to different categories
(typically food, service, atmosphere, etc.). Qualitative rating on the other
hand gathers all these various features in a single text.  
In this project, we aim at extracting the relevant snippets of a review
for each category we are interested in. Various levels of granularity
can be implemented. Using this categorization, we can highlight the most
prominent qualities reported in the reviews for each category.
Another aspect of qualitative reviews is sentiment. After categorizing
the snippets, we can extract sentiment to enrich the quantitative rating.


#### Motivation
The information burried in written reviews is a still under utilized bounty,
both by potential customers and reviewed businesses. What we propose here
is a way to take better advantage of reviews to refine the fast digestion
of both quantitative and qualitative restaurant reviews.  
Furthermore, sentiment analysis enables us to better contextualize quantitative
rating.


<!-- #### Deliverables
The final deliverable of this project is a UI that will provide users with
a contextualization of numeric ratings in terms of pregnant features within
categories and salient associated sentiments.
 -->

#### Data sources
For this project, we use data from [Open Table] (http://www.opentable.com),
which has detailled rating. Next, using [Yelp] (http://www.yelp.com) reviews,
we calculate the detailled rating based on the sentiment analysis benchmarked
with Open Table reviews.


#### Description
This is still under construction.

##### Scraping
This module (left during the construction phase) is used to aquire data
from [Open Table] (http://www.opentable.com).

##### Extracting data
Extracting the data from MongoDB and transforming it in a usable format.

##### Exploratory Data Analysis (EDA)
Gathering basic statistics on the reviews.

##### Processing data
The core of the work: TFIDF vectorization, NMF factorization to extract
latent topics in the reviews, (so far) hand-labelling and grouping them
by category, calling sentiment analysis module.

##### Sentiment analysis
Still very rough

<!-- ### Process
A high-level description of the investigation process is the following:
1.	Acquiring the data, explore, and clean it.  
	In particular, very short reviews will be discarded to reduce noise.
	Subdivision of the data may be useful: we will assess the relevance
	of this practice.
2.	Using the set of reviews hence extracted, we will perform topic
	extraction (e. g. with non-negative matrix factorization).
3.	Labelling the extracted topics into categories of interest.
4.	Parse the review into snippets (the precise definition of which
	will be evaluated) and calculate the weight vector of each snippet
	in the topic space.
5.	Extracting for each category the most prominent features hierarchically
	ordered alng the granularity deduced in 2.
6.	Further proceed to a sentiment analysis on each snippet, hence producing
	a qualitative rating.
7.	Building a UI to show results (e. g. using Flask)

Nota: 3-4-5-7 can be iteratively carried out to deliver intermediate working
products. 6 can be added after a few iterations and 6-7 can then be further
improved. The leading principle is to regularly deliver working products
along the development process. -->
