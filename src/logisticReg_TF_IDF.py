from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def create_logistic_reg_tfidf_model():
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
            "logreg",
            LogisticRegression(
                max_iter=1000,           
                class_weight="balanced", 
                random_state=42,       
            ),
        ),
    ])
    return model
