from pprint import pprint
from time import time
import logging
from tqdm import tqdm

import pandas as pd
import numpy as np
import nltk
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')

def loadData(pathFileCompany1,pathFileCompany2,pathFileGroundTruth):
    logging.info("Load company1 data")
    company1 = pd.read_csv(pathFileCompany1)
    company1['Company'] = "company1"
    logging.info(f"#Record company1 : {len(company1)}")
    logging.info("Load company2 data")
    company1 = company1.rename(columns={"title": "name"})
    company2 = pd.read_csv(pathFileCompany2)
    company2['Company'] = "company2"
    logging.info(f"#Record company2 : {len(company2)}")
    logging.info("Load groundtruth data")
    GroundTruth = pd.read_csv(pathFileGroundTruth)
    logging.info(f"#Record groundtruh : {len(GroundTruth)}")
    return company1,company2,GroundTruth

class NLTKprocessing(BaseEstimator, ClassifierMixin):
    def __init__(self,ngram_range=None,max_df=None):
    #def __init__(self, max_df=None):
        print(">>>>>>>>NLTKProcessing->init() called.")
        #self.ngram_range = ngram_range
        self.ngram_range = ngram_range
        self.max_df = max_df

    def TFIDF(self,X,Y=None):
        print(">>>>>>>>NLTKProcessing->TFIDF() called.")
        #vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.10)  # ngram_range=(1),
        vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, max_df=self.max_df)
        vectors = vectorizer.fit_transform(X['full data'])
        feature_names = vectorizer.get_feature_names()
        return vectors.todense()

    def prep(self,texte):
        #print(">>>>>>>>NLTKProcessing->text() called.")
        # suppression des caracteres non alphanumériques + tout en minuscule
        texte = re.sub("[^a-zA-Z0-9_]", " ", str(texte)).lower()
        # tokenization par mot
        tokens = nltk.word_tokenize(texte)
        # supreesion des stopwords
        filtered_tokens = [w for w in tokens if not w in stop_words]
        #    # Stemming
        #    texte = [nltk.stem.SnowballStemmer('english').stem(w) for w in filtered_tokens]
        # Lemmatization
        texte = [nltk.stem.WordNetLemmatizer().lemmatize(w) for w in filtered_tokens]
        # remise sous forme d'une string
        return " ".join(texte)

    #def fit(self,df,**fit_params):
    def fit(self, company1,company2):
        print(">>>>>>>>NLTKProcessing->fit() called.")
        print(f"len(company1)={len(company1)}")
        print(f"len(company2)={len(company2)}")
        corpus = pd.concat([company1, company2], sort=False, ignore_index=True)
        corpus['name'] = corpus['name'].fillna(' ')
        corpus['manufacturer'] = corpus['manufacturer'].fillna(' ')
        corpus['description'] = corpus['description'].fillna(' ')
        corpus['full data'] = corpus['manufacturer'].apply(self.prep) + ' ' + corpus['name'].apply(self.prep)
        number_of_matches = 0
        self.matches = []
        dense = self.TFIDF(corpus)
        for i in range(len(company1)):
            try:
                price1 = float(company1.iloc[i, 4])
            except:
                price1 = 0
            for j in range(len(company2)):
                try:
                    price2 = float(company2.iloc[j, 4])
                except:
                    price2 = 0
                if price1 * price2 == 0:
                    price_ratio = 1
                else:
                    price_ratio = max(price1, price2) / min(price1, price2)
                similarity = np.dot(dense[i], np.transpose(dense[len(company1) + j])).item(0)
                if ((similarity > 0.5) and (price_ratio < 2)):  # or name_score<=1) :
                    number_of_matches = number_of_matches + 1
                    self.matches.append((company1.iloc[i, 0], company2.iloc[j, 0]))
                    #self.matches.append(str(company1.iloc[i, 0]) + "-" + str(company2.iloc[j, 0]))
        print("Number of matches: {}".format(number_of_matches))
        return self.matches

    def predict(self,ground_truth_matches):
        print(">>>>>>>>NLTKprocessing->predict() called.")
        # Check is fit had been called
        #check_is_fitted(self)
        matches_df = pd.DataFrame(self.matches)
        matches_df.columns = ['idCompany1', 'idCompany2']
        diff_df = pd.merge(ground_truth_matches, matches_df, how='outer', indicator='Exist')
        true_positives = diff_df[diff_df.Exist == 'both']
        false_positives = diff_df[diff_df.Exist == 'right_only']
        false_negatives = diff_df[diff_df.Exist == 'left_only']
        print("Number of true positives: {}".format(len(true_positives)))
        print("Number of false positives: {}".format(len(false_positives)))
        print("Number of false negatives: {}".format(len(false_negatives)))
        precision = len(true_positives) / (len(true_positives) + len(false_positives))
        print("Precision: {}".format(precision))
        recall = len(true_positives) / (len(true_positives) + len(false_negatives))
        print("Recall: {}".format(recall))
        f_measure = 2 * (precision * recall) / (precision + recall)
        print("F measure: {}".format(f_measure))
        toto = 0
        return toto

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

if __name__ == "__main__":
    logging.info("Start")
    pathFileCompany1 = "./SampleData/Sample_Company1.csv"
    pathFileCompany2 = "./SampleData/Sample_Company2.csv"
    pathFileGroundTruth = "./SampleData/Sample_Groud_truth_mappings.csv"

    stop_words = set(nltk.corpus.stopwords.words('english'))
    stop_words.update("r")


    df_load = loadData(pathFileCompany1,pathFileCompany2,pathFileGroundTruth)
    company1 = df_load[0]
    company2 = df_load[1]
    GroundTruth = df_load[2]
    #tmp = GroundTruth['idCompany1']+"-"+"idCompany2"


    model=NLTKprocessing((1,2),0.10)
    Y_pred = model.fit(company1,company2)
    #accuracy_score(Y_pred,tmp)
    model.predict(GroundTruth)


    '''parameters = {"ngram_range": [(1, 2)],"max_df": [0.1]}
    #parameters = {"max_df": [0.1,0.2]}
    grid = GridSearchCV(NLTKprocessing,parameters)
    grid.fit(company1, company2)'''


    '''parameters = {"ngram_range": (1,2), "max_df": [1]}
    pipe = Pipeline([
        ('tfidf',TfidfVectorizer(parameters)),
        ('to_dense',ToDenseTransformer)
    ])
    #vectors = pipe.fit_transform(corpus['full data'])
    #dense = vectors.todense()
    pipe.fit_transform(corpus['full data'])
    '''
    '''grid_search_tune = GridSearchCV(pipe, parameters, cv=2, n_jobs=2, verbose=3)
    grid_search_tune.fit(corpus)'''

    '''class ToDenseTransformer():
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        return X.todense()
    '''