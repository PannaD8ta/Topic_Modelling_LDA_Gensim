# Topic Modelling Analysis with latent Dirichlet allocation (LDA): Project Overview

As a whole, topic modelling refers to the technique of the algorithm (Unsupervised machine learning method) to discover huge volume of data generated in low dimension and to uncover hidden concepts, prominent features or latent variables of data, depending on the application of concept. Examples of real-world data includes social media feeds, product customer reviews, user feedbacks, tweets, e-mails, spams, and customer complaints. Initially - through the algebraic perspective, dimension reduction can be seen to decomponse the original matrix into factor matrix. Hence, the typical classification strategy can be described as probablistic models and non-probabilistic models. 

In this mini-project, I attempt to identify the topics based on a procurement dataset using a topic modelling technique (LDA). 

## Other Use-Cases
### 1. Customer Service
+ Tagging automation system of customer support tickets based on topic or the recognition of patterns, which results in the form of words or expressions that occured regularly.
+ Automatically dividing, prioritizing, and routing conversation to the most appropriate team.
+ Obtaining insights from customer support conversations

### 2. Customer Feedback
+ Classifying and modelling topics from customer's feedback via review, social media posts, emails, chats and surveys and responding methodically and strategically that will make customers want to use a company's services or products again. 

## Resources Used
**Programming language:** Python 3.7

**Packages:** pandas, numpy, spaCy, tqdm, os, re, operator, gensim, nltk, pyLDAvis, wordcloud, matplotlib.

## Dataset
- [Kaggle](https://www.kaggle.com/nikhil1011/product-category-from-invoice/data)

- Data points:-
  - Category (categorical) 
  - Inv_Id (integer)
  - Vendor_Code (string)
  - Item_Description (string)
  - Product_Category (integer)

## Data Pre-processing

For our data and analysis, this stage will be divided into the following steps:

- Remove quotation marks
- Remove punctuations
- Tokenization
- Make Bigrams & Trigrams
- Stop words removal: using NLTK
- Remove words less than 2 characters
- Lemmatization: using spaCy

## EDA

## Model Building

## Model Performance

## References:
- https://github.com/topics/topic-modeling
- https://pyldavis.readthedocs.io/
- https://spacy.io/usage/spacy-101
- https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21


