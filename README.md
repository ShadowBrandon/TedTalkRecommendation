# TEDTalkRecommendation

Ted Talk Recommender

This repo contains Ipython/Jupyter notebooks for basic exploration of transcripts of Ted Talks using Natural Language Processing (NLP), topic modeling, and a recommender that lets you enter key words from the title of a talk and finds 5 talks that are similar. The data consists of transcripts from Ted and TedX talks. https://www.kaggle.com/rounakbanik/ted-talks

The initial cleaning and exploration are done in

TedTalk_Explore.ipynb

Cleaning Text with NLTK

Four important steps for cleaning the text and getting it into a format that we can analyze: 1)tokenize 3)remove stop words/punctuation 4)vectorize

NLTK (Natural Language ToolKit) is a python library for NLP. I found it very easy to use and highly effective.

tokenize- the process of splitting up the document (talk) into words. There are a few tokenizers in NLTK, and one called wordpunct was my favorite because it separated the punctuation as well.

Remove stop words/punctuation - the moving words that aren't essential words contributing to the message of a TED talk like commas, periods, articles, pronouns, etc.

Vectorization - turning our words into numbers. The method that gave me the best results was count vectorizer. This function takes each word in each document and counts the number of times the word appears. You end up with each word as your columns and each row is a document (talk), so the data is the frequency of each word in each document. As you can imagine, there will be a large number of zeros in this matrix; we call this a sparse matrix.

Topic modeling!

First get the cleaned_talks from the previous step. Then import the models

    from sklearn.decomposition import LatentDirichletAllocation,  TruncatedSVD, NMF

We will try each of these models and tune the hyperparameters to see which one gives us the best topics (ones that make sense to you). It's an art.

This is the main format of calling the model, but I put it into a function along with the vectorizers so that I could easily manipulate the paremeters like 'number of topics, number of iterations (max_iter),n-gram size (ngram_min,ngram_max), number of features (max_df):

    lda = LatentDirichletAllocation(n_components=topics,
                                        max_iter=iters,
                                        random_state=42,
                                        learning_method='online',
                                        n_jobs=-1)
       
    lda_dat = lda.fit_transform(vect_data)

The functions will print the topics and the most frequent 15 words in each topic.

The best parameter to tweak is the number of topics, higher is more narrow, but I decided to stay with a moderate number (15) because I didn't want the recommender to be too specific in the recommendations.

Once we get the topics that look good, we can do some clustering to improve it further. However, as you can see, these topics are already pretty good, so we will just assign the topic with the highest score to each document.

    topic_ind = np.argmax(lda_data, axis=1)
    topic_ind.shape
    y=topic_ind

Then, you have to decide what to name each topic. Do this and save it for plotting purposes in topic_names. Remember that LDA works by putting all the noise into one topic, so there should be a 'junk' topic that makes no sense. I realize that as you look at my code, you will see that I have not named a 'junk' topic here. The closest was the 'family' topic but I still felt like it could be named. Usually, when running the models with a higher number of topics (25 or more) you would see one that was clearly junk.

        topic_names = tsne_labels
        topic_names[topic_names==0] = \"family\" 
        . . .

Then we can use some visualization tools to 'see' what our clusters look like. The pyLDAviz is really fun, but only plots the first 2 components, so it isn't exactly that informative. I like looking at the topics using this tool, though. Note: you can only use it on LDA models.

The best way to 'see' the clusters, is to do another dimensionality reduction and plot them in a new (3D) space. This is called tSNE (t-Distributed Stochastic Neighbor Embedding. When you view the tSNE ,it is important to remember that the distance between clusters isn't relevant, just the clumpiness of the clusters. For example, do the points that are red clump together or are they really spread out? If they are all spread out, then that topic is probably not very cohesive (the documents in there may not be very similar).

After the tSNE plot, you will find the functions to run the other models (NMF, Truncated SVD).



Recommender System:

tedtalk.py - 

Load the entire data set, and all the results from the LDA model. The function will take in a talk (enter the ID number) and find the 3 closest talks using nearest neighbors.

The distance, topic name, url, and ted's tags for the talk will print for the talk you enter and each recommendation.
