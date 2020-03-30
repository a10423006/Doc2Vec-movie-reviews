# Use Google's Doc2Vec for movie reviews
### Kaggle Challenge: [Bag of words meets bags of popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial/overview)

## Tutorial Overview 
>  This tutorial will help you get started with Word2Vec for natural language processing. It has two goals:   
>  
> **Basic Natural Language Processing**: Part 1 of this tutorial is intended for beginners and covers basic natural language processing techniques, which are needed for later parts of the tutorial.  
>  
> **Deep Learning for Text Understanding**: In Parts 2 and 3, we delve into how to train a model using Word2Vec and how to use the resulting word vectors for sentiment analysis.  
>  
> Since deep learning is a rapidly evolving field, large amounts of the work has not yet been published, or exists only as academic papers. Part 3 of the tutorial is more exploratory than prescriptive -- we experiment with several ways of using Word2Vec rather than giving you a recipe for using the output.
>  
> To achieve these goals, we rely on an IMDB sentiment analysis data set, which has 100,000 multi-paragraph movie reviews, both positive and negative. 

簡單來說就是使用 _Word2Vec_ 對 IMDB 的電影評論進行情感分析，透過模型判斷該評論為正面或負面。

雖然網頁上是要求使用 **_Word2Vec_**，但我是使用它的延伸應用 **_Doc2Vec_**
[models.doc2vec – Doc2vec paragraph embeddings](https://radimrehurek.com/gensim/models/doc2vec.html)

## Data Set
---
訓練資料和測試資料各 25000筆，另包含沒有標註情感分數的額外訓練資料集 50000筆
 
![訓練資料集欄位](https://github.com/a10423006/Doc2Vec-movie-reviews/blob/master/images/image_1.png)

![訓練資料集情感分佈](https://github.com/a10423006/Doc2Vec-movie-reviews/blob/master/images/sentiment_bar.png)

---
| | SVC | Decision Tree | Random Forest | Logistic Regression | KNN |
|:--------:|:-----:|:------:|:-----:|:-----:|:-----:|
| **Accuracy** | <font color="red">0.837</font> | 0.639 | 0.808 | 0.833 | 0.709 |
| **MSE** | 0.159 | 0.0 | 0.0 | 0.163 | 0.202 |
| **MAE** | 0.159 | 0.0 | 0.0 | 0.163 | 0.202 |

![SVC 模型 ROC需曲線圖](https://github.com/a10423006/Doc2Vec-movie-reviews/blob/master/images/roc.png)


