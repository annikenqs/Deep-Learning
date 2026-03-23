from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def create_naive_bayes_tfidf_model():
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,         
                stop_words="english", 
                max_features=50000, 
                ngram_range=(1, 2),     
            ),
        ),
        (
            "nb",
            MultinomialNB()          
        ),
    ])
    return model