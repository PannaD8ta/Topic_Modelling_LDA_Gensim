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

**Packages:** pandas, numpy, spaCy, tqdm, os, re, operator, gensim, nltk, pyLDAvis, wordcloudy, matplotlib.

## Dataset
- Source: [Kaggle](https://www.kaggle.com/nikhil1011/product-category-from-invoice/data)

- Data points:-
  - Category (categorical) 
  - Inv_Id (integer)
  - Vendor_Code (string)
  - Item_Description (string)
  - Product_Category (integer)

## 1. Data Pre-processing

For our data and analysis, this stage will be divided into the following steps:

- Remove quotation marks
- Remove punctuations
- Tokenization
- Make Bigrams & Trigrams
- Stop words removal: using NLTK
- Remove words less than 2 characters
- Lemmatization: using spaCy

### Text Cleaning

The following function was used to clean the text and return a list of tokens:

<p float="left">
  <img src="https://github.com/PannaD8ta/Topic_Modelling_LDA_Gensim/blob/main/img/Pre-processing.png" alt="code snippet" width="450" height="450"/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>

Now we can see the conversion of the data after the process:

**[['agency', 'fees', 'commissions', 'production', 'jun', 'ames', 'department', 'stores', 'inc', 'smap', 'other', 'agency']**

**['miscellaneous', 'company', 'car', 'field', 'only', 'aug', 'travel', 'and', 'entertainment', 'air', 'products', 'chemicals', 'inc', 'ground', 'transportation', 'miscellaneous', 'company', 'car', 'field', 'only']**

**['real', 'estate', 'store', 'management', 'lease', 'rent', 'base', 'rent', 'may', 'ferguson', 'marshall']**

**['corning', 'inc', 'auto', 'leasing', 'sep', 'auto', 'leasing', 'and', 'maintenance', 'other', 'corporate', 'services', 'corporate', 'services']** 

**['ground', 'transportation', 'travel', 'and', 'entertainment', 'miscellaneous', 'company', 'car', 'field', 'only', 'miscellaneous', 'company', 'car', 'field', 'only', 'mar', 'butler', 'manufacturing']]**


### Bigrams & Trigrams Models

The use of bigrams and trigrams can increase the quality of feature sets, as some of the words can be a two-word or three-word sequence of words. The following function was used for bigrams and trigrams.

<p float="left">
  <img src="https://github.com/PannaD8ta/Topic_Modelling_LDA_Gensim/blob/main/img/bigramtrigram3.png" alt="code snippet" width="800" height="550"/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>

## 2. Using Gensim for LDA modelling

The code creates a Dictionary from the data, then converted to a "bag-of-words" corpus with the purpose of saving the dictionary and corpus for later on. 

<p float="left">
  <img src="https://github.com/PannaD8ta/Topic_Modelling_LDA_Gensim/blob/main/img/dictionarycorpus.png" alt="code snippet" width="500" height="450"/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>
<p float="left">
  <img src="https://github.com/PannaD8ta/Topic_Modelling_LDA_Gensim/blob/main/img/buildldamodel.png" alt="code snippet" width="1800" height="250"/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>

## 3. Model Evaluation
- Why do we need to evaluate the model?
  - has it captured the internal structure of the corpus?
  - are the topic undertandable?
  - are the topic coherent?
  - does it serve the purpose it is being used for:
    - the use case defines a "good" model
    - often there is no objective good
- Topic models are notoriously difficult to evaluate
- Qualitative evaluation is labour intensive and often subjective. 
  - Intrinsic methods that do not always produce easy to understand results
  - Inter-annotator agreement on what is understandable (datasets are expensive to produce)
  - sets of words that capture the semantics of a category
- A universally good model doesn't exists, depends on what you want to do with the model

Now we will define a function to measure the topic coeherence:

<p float="left">
  <img src="https://github.com/PannaD8ta/Topic_Modelling_LDA_Gensim/blob/main/img/topiccoherence.png" alt="code snippet" width="1800" height="450"/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>

### No. of Topics vs Coherence Score:
- The improvement stops significantly improving after 13 topics. It is not always best where the highest Cv is, so we can try multiple to find the best result. Adding topics can help reveal further sub topics. Nonetheless, if the same words start to appear across multiple topics, the number of topics is too high.
- Coherence Score was recorded at 0.49483701304480177

<p float="left">
  <img src="https://github.com/PannaD8ta/Topic_Modelling_LDA_Gensim/blob/main/img/coherenceplot.png" alt="code snippet" width="700" height="450"/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>

## 4. Initial Model & Visualisation
- pyLDAvis is a an efficient package that helps to interpret the topics in a topic model that has been fitted to a corpus of text data and it helps extract information from a fitted LDA topic model into an interactive web-based visualisation. 
- Many overlapping topics from the visualisation below. We need to try to have topics to be as independent as possible. We will try to optimize the model next.
- Saliency: a measure of how much the term tells you about the topic.
- Relevance: a weighted average of the probability of the word given the topic and the word given the topic normalized by the probability of the topic.
- The size of the bubble measures the importance of the topics, relative to the data.

<p float="left">
  <img src="https://github.com/PannaD8ta/Topic_Modelling_LDA_Gensim/blob/main/img/visualisation.png" alt="code snippet" width="700" height="450"/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>

## 5. Optimised Model & Visualisation

First, we find the optimal number of topics for LDA based on the information below:

- If the training corpus has 200 documents, chunksize is 100, passes is 2, and iterations is 10, algorithm goes through these rounds.
  - Round #1: documents 0–99
  - Round #2: documents 100–199
  - Round #3: documents 0–99
  - Round #4: documents 100–199
- Each round will iterate each document’s probability distribution assignments for a maximum of 10 times, moving to the next document before 10 times if it already reached convergence.
- With 5500 documents, we will estimate the chunksize is 2750, passes is 550, and iterations is 275. This information will be entered manually in the topic coherence function

### Optimised No. of Topics vs Coherence Score plot:
<p float="left">
  <img src="https://github.com/PannaD8ta/Topic_Modelling_LDA_Gensim/blob/main/img/optimisedcoherenceplot.png" alt="code snippet" width="700" height="450"/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>
- Coherence Score was recorded at 0.5035499329043246, which was not significant in improvement. 
<br>
<br>

### Optimised Visualisation:

- As u can see from the visualisation below, the optimised version shows an improvement with having the topics to be more independent. 

<p float="left">
  <img src="https://github.com/PannaD8ta/Topic_Modelling_LDA_Gensim/blob/main/img/visualisation2.png" alt="code snippet" width="700" height="450"/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>

## References:
- [ ] Blei et. al, *Latent Dirichlet Allocation*, University of California, 2003 [link](https://ai.stanford.edu/~ang/papers/nips01-lda.pdf)
- [ ] Blei. D, *Probalistic Topic Models and User Behaviour*, University of Edinburgh 2017 [link](https://www.youtube.com/watch?v=FkckgwMHP2s)
- [ ] Newman et. al *Automatic Evaluation of Topic Coherence*, HLT 2010 [link](https://dl.acm.org/doi/10.5555/1857999.1858011)
- [ ] Mimno et. al *Optimizing Semantic Coherence in Topic Models*, EMNLP 2011 [link](http://dirichlet.net/pdf/mimno11optimizing.pdf)
- [ ] Aletras et. al *Evaluating Topic Coherence Using Distributional Semantics*, IWCS 2013 [link](https://www.aclweb.org/anthology/W13-0102.pdf)
- [ ] Röder et. al *Exploring the Space of Topic Coherence Measures*, WSDM 2015
 [link](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf)
- [ ] Ruchirawat N., *6 Tips for Interpretable Topic Models*, towardsdatascience 2020 [link](https://towardsdatascience.com/6-tips-to-optimize-an-nlp-topic-model-for-interpretability-20742f3047e2)
- [ ] Ruchirawat N., *Collocations — identifying phrases that act like single words in Natural Language Processing* [link](https://medium.com/@nicharuch/collocations-identifying-phrases-that-act-like-individual-words-in-nlp-f58a93a2f84a)
- [ ] Lyra M., *Evaluating Topic Models*, PyData Berlin 2017 [link](https://www.youtube.com/watch?v=UkmIljRIG_M&t)
- [ ] https://github.com/topics/topic-modeling
- [ ] https://pyldavis.readthedocs.io/
- [ ] https://spacy.io/usage/spacy-101
- [ ] https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21


