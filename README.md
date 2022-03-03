# Naive Bayes
## Project Description
As part of a project for our Univesity lesson "Artificial Intelligence" we implemented the Naive Bayes algorithm, in order to classify different reviews as positive or negative.
For the data we used the ["Large Movie Review Dataset"](https://ai.stanford.edu/~amaas/data/sentiment/), also known as ["IMDB dataset"](https://keras.io/api/datasets/imdb/).

## Details
Each piece of text (review) was represented as an array of 1s and 0s, that show which words are used in it from a vocabulary.
The vocabulary mentioned includes the m most used words from the training data, excluding the 50 most used ones.
In the end, the results can be compared with implemention provided by Scikit-learn.
