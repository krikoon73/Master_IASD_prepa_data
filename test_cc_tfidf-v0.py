from pprint import pprint
from time import time
import logging
from tqdm import tqdm

import pandas as pd
import numpy as np
import nltk
import re
import math

from matplotlib import pyplot as plt

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
# from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
# from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score

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
    #def __init__(self,similarity_threshold=None,ngram_range=None,max_df=None):
    #def __init__(self, parameters=None):
    def __init__(self, similarity_threshold, ngram_range, max_df):
        logging.debug(">>>>>>>>NLTKProcessing->init() called.")
        self.similarity_threshold = similarity_threshold
        #self.similarity_threshold = parameters[0]
        self.ngram_range = ngram_range
        #self.ngram_range = parameters[1]
        self.max_df = max_df
        #self.max_df = parameters[2]
        logging.debug(f"similarity_threshold={self.similarity_threshold}")
        logging.debug(f"ngram_range={self.ngram_range}")
        logging.debug(f"max_df={self.max_df}")

    def TFIDF(self,X,Y=None):
        logging.debug(">>>>>>>>NLTKProcessing->TFIDF() called.")
        logging.debug(f"ngram_range={self.ngram_range}")
        logging.debug(f"max_df={self.max_df}")
        #vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.10)  # ngram_range=(1),
        vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, max_df=self.max_df,sublinear_tf=True)
        vectors = vectorizer.fit_transform(X['full data'])
        feature_names = vectorizer.get_feature_names()
        return vectors.todense()

    def prep(self,texte):
        logging.debug(">>>>>>>>NLTKProcessing->text() called.")
        # suppression des caracteres non alphanumériques + tout en minuscule
        texte = re.sub("[^a-zA-Z0-9_]", " ", str(texte)).lower()
        # tokenization par mot
        tokens = nltk.word_tokenize(texte)
        # supression des stopwords
        filtered_tokens = [w for w in tokens if not w in stop_words]
        #    # Stemming
        #    texte = [nltk.stem.SnowballStemmer('english').stem(w) for w in filtered_tokens]
        # Lemmatization
        texte = [nltk.stem.WordNetLemmatizer().lemmatize(w) for w in filtered_tokens]
        # remise sous forme d'une string
        return " ".join(texte)
    #def test_recursion(self,dense,df_a,df_b,i,j,n1,n2,matches,number_of_matches):
    def test_recursion(self, i, j, n1, n2, matches, number_of_matches):
        logging.debug(">>>>>>>>NLTKProcessing->test_recursion !!")
        logging.debug(f"i={i} j={j}")
        '''if i == 0:
            logging.debug("INIT matches and number_of_matches")
            matches = []
            number_of_matches = 0'''
        if i > n1:
            logging.info("!! STOP RECURSION !!")
            logging.info("Number of matches: {}".format(number_of_matches))
            self.matches = matches
            return 0
        j = 0
        try:
            #price1 = float(df_a.iloc[i, 4])
            price1 = float(self.df_a.iloc[i, 4])
        except:
            price1 = 0
        while (i <= n1 and j <= n2):
            logging.debug(f"IN WHILE 1 - i={i} j={j}")
            try:
                #price2 = float(df_b.iloc[j, 4])
                price2 = float(self.df_b.iloc[j, 4])
            except:
                price2 = 0
            if price1 * price2 == 0 or max(price1, price2) / min(price1, price2) < 2:
                try:
                    #similarity = np.dot(dense[i],np.transpose(dense[len(df_a) + j])).item(0)/math.sqrt(np.dot(dense[i],np.transpose(dense[i])).item(0) * np.dot(dense[len(df_a) + j],np.transpose(dense[len(df_a) + j])).item(0))
                    similarity = np.dot(self.dense[i], np.transpose(self.dense[len(self.df_a) + j])).item(0) / math.sqrt(
                        np.dot(self.dense[i], np.transpose(self.dense[i])).item(0) * np.dot(self.dense[len(self.df_a) + j], np.transpose(
                            self.dense[len(self.df_a) + j])).item(0))
                except:
                    similarity = 0
                if ((similarity > self.similarity_threshold)):  # or name_score<=1) :
                    number_of_matches = number_of_matches + 1
                    matches.append((self.df_a.iloc[i, 0], self.df_b.iloc[j, 0]))
            j += 1
            logging.debug(f"IN WHILE 2 - i={i} j={j}")
        i += 1
        #self.test_recursion(dense,df_a,df_b,i,j,n1,n2,matches,number_of_matches)
        self.test_recursion(i, j, n1, n2, matches, number_of_matches)
    #def fit(self,df,**fit_params):
    def fit(self, company1,company2):
        logging.debug(">>>>>>>>NLTKProcessing->fit() called.")
        logging.debug(f"len(company1)={len(company1)}")
        logging.debug(f"len(company2)={len(company2)}")
        corpus = pd.concat([company1, company2], sort=False, ignore_index=True)
        corpus['name'] = corpus['name'].fillna(' ')
        corpus['manufacturer'] = corpus['manufacturer'].fillna(' ')
        corpus['description'] = corpus['description'].fillna(' ')
        corpus['full data'] = corpus['manufacturer'].apply(self.prep) + ' ' + corpus['name'].apply(self.prep)
        number_of_matches = 0
        self.matches = []
        dense = self.TFIDF(corpus)
        self.matches = []
        number_of_matches = 0
        self.dense = dense
        #self.df_a = company1
        #self.df_b = company2
        #self.test_recursion(dense,company1,company2,0,0,len(company1)-1,len(company2)-1,self.matches,number_of_matches)
        #self.test_recursion(0, 0, len(company1) - 1, len(company2) - 1, self.matches,number_of_matches)
        n1 = len(company1)
        n2 = len(company2)
        #outer_bar = tqdm(range(n1), desc = 'Loop on company1')
        #inner_bar = tqdm(range(n2), desc = 'Loop on company2')
        for i in tqdm(range(n1),position=0, desc="Loop on company1", leave=False, colour='green', ncols=80):
            #outer_bar.update(1)
            try:
                price1 = float(company1.iloc[i, 4])
            except:
                price1 = 0
            for j in tqdm(range(n2),position=1, desc="Loop on company2", leave=False, colour='red', ncols=80):
                #inner_bar.update(1)
                try:
                    price2 = float(company2.iloc[j, 4])
                except:
                    price2 = 0
                if price1 * price2 == 0 or max(price1, price2) / min(price1, price2) < 2:
                    try:
                        similarity = np.dot(dense[i],
                                            np.transpose(dense[len(company1) + j])).item(0) \
                                     / math.sqrt(np.dot(dense[i],
                                                        np.transpose(dense[i])).item(0) * np.dot(dense[len(company1) + j],
                                                                                      np.transpose(dense[len(
                                                                                          company1) + j])).item(0))
                    except:
                        similarity = 0
                    if ((similarity > self.similarity_threshold)):  # or name_score<=1) :
                        number_of_matches = number_of_matches + 1
                        self.matches.append((company1.iloc[i, 0], company2.iloc[j, 0]))
                #inner_bar.reset()
        logging.debug("Number of matches: {}".format(number_of_matches))
        return self.matches

    def predict(self,ground_truth_matches):
        logging.debug(">>>>>>>>NLTKprocessing->predict() called.")
        # Check is fit had been called
        #check_is_fitted(self)
        matches_df = pd.DataFrame(self.matches)
        matches_df.columns = ['idCompany1', 'idCompany2']
        diff_df = pd.merge(ground_truth_matches, matches_df, how='outer', indicator='Exist')
        true_positives = diff_df[diff_df.Exist == 'both']
        false_positives = diff_df[diff_df.Exist == 'right_only']
        false_negatives = diff_df[diff_df.Exist == 'left_only']
        logging.debug("Number of true positives: {}".format(len(true_positives)))
        logging.debug("Number of false positives: {}".format(len(false_positives)))
        logging.debug("Number of false negatives: {}".format(len(false_negatives)))
        precision = len(true_positives) / (len(true_positives) + len(false_positives))
        logging.debug("Precision: {}".format(precision))
        recall = len(true_positives) / (len(true_positives) + len(false_negatives))
        logging.debug("Recall: {}".format(recall))
        f_measure = 2 * (precision * recall) / (precision + recall)
        logging.debug("F measure: {}".format(f_measure))
        return true_positives,false_positives,false_negatives,precision,recall,f_measure,diff_df

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":
    logging.info("Start")
    #pathFileCompany1 = "./SampleData/Sample_Company1.csv"
    #pathFileCompany2 = "./SampleData/Sample_Company2.csv"
    #pathFileGroundTruth = "./SampleData/Sample_Groud_truth_mappings.csv"

    pathFileCompany1 = "./Data/Company1.csv"
    pathFileCompany2 = "./Data/Company2.csv"
    pathFileGroundTruth = "./Data/Ground_truth_mappings.csv"

    stop_words = set(nltk.corpus.stopwords.words('english'))
    stop_words.update("r")

    df_load = loadData(pathFileCompany1,pathFileCompany2,pathFileGroundTruth)
    company1 = df_load[0]
    company2 = df_load[1]
    GroundTruth = df_load[2]

    score = pd.DataFrame(columns=['nb_ref_company1',
                                  'nb_ref_company2',
                                  'ngram_range',
                                  'max_df',
                                  'similarity_threshold',
                                  'true_positives',
                                  'false_positives',
                                  'false_negatives',
                                  'precision',
                                  'recall',
                                  'f_measure',
                                  'processing_time'])
    ngram_range = (1,2)
    max_df = 0.1
    Min = 0.
    Max = 1
    step = 0.05
    s = 0
    epoch = 0
    epochs = round((Max-Min)/step)
    #pbar = tqdm(total=epochs,position=1)
    #for s in tqdm(np.arange(Min,Max,step)):
    t_start = time()
    while s < Max:
        epoch +=1
        logging.info(f"epoch={epoch} - similiraty_threshold={round(s, 2)} - START")
        t_start_model = time()
        model=NLTKprocessing(s,ngram_range,max_df)
        Y_pred = model.fit(company1,company2)
        results = model.predict(GroundTruth)
        t_stop_model = time()
        true_positives = results[0]
        false_positives = results[1]
        false_negatives = results[2]
        precision = results[3]
        recall = results[4]
        f_measure = results[5]
        score = score.append({'nb_ref_company1':len(company1),
                              'nb_ref_company2':len(company2),
                              'ngram_range':ngram_range,
                              'max_df':max_df,
                              'similarity_threshold': s,
                              'true_positives':len(true_positives),
                              'false_positives':len(false_positives),
                              'false_negatives':len(false_negatives),
                              'precision':precision,
                              'recall':recall,
                              'f_measure':f_measure,
                              'processing_time':t_stop_model-t_start_model},
                             ignore_index=True)
        logging.info(f"epoch={epoch} - similiraty_threshold={round(s, 2)} - time = {t_stop_model-t_start_model} / f_measure={f_measure}")
        s += step
        #pbar.update(1)
    t_stop = time()
    logging.info(f"PROCESSING_TIME={t_stop-t_start}")
    #results[6]

    score['tpr'] = score['true_positives'] / (score['true_positives']+score['false_negatives'])
    nb_distinct_ref_company1 = len(pd.unique(company1['id']))
    logging.debug(f"nb_distinct_ref_company1={nb_distinct_ref_company1}")
    nb_distinct_ref_company2 = len(pd.unique(company2['id']))
    logging.debug(f"nb_distinct_ref_company2={nb_distinct_ref_company2}")
    #score['true_negatives'] = abs(score['true_positives']-min(nb_distinct_ref_company1,nb_distinct_ref_company2))
    score['true_negatives'] = nb_distinct_ref_company1*nb_distinct_ref_company2 - score['true_positives'] - score['false_positives'] - score['false_negatives']


    score['fpr'] = score['false_positives']/(score['true_negatives']+score['false_positives'])
    filename_score = "results/scores_run-full.csv"
    score.to_csv(filename_score,index=False)

    fig,ax1 = plt.subplots(1,1,figsize=(8,4))

    score.plot(ax=ax1,kind='line', x='fpr', y='tpr', label='tpr',legend=True)
    ax2 = ax1.twinx()
    ax2 = score.plot(secondary_y=True,
               x='fpr',
               y='similarity_threshold',
               label='similarity_threshold',
               ax=ax2,
               kind='scatter',
               marker='+',
               color='orange',legend=True)
    ax1.set_ylabel('tpr')
    ax2.set_ylabel("similiraty_threshold")
    ax1.legend(loc="center")
    ax2.legend(loc="center right")

    plt.title("ROC")
    #plt.show()
    filename_roc = "results/ROC_run-full.png"
    plt.savefig(filename_roc,bbox_inches='tight')
    plt.close(fig)


    '''model=NLTKprocessing(0.2,(1,2),0.10)
    Y_pred = model.fit(company1,company2)
    model.predict(GroundTruth)'''



    '''#parameters = {"similarity_threshold":[0.2],"ngram_range": [(1, 2)], "max_df": [0.1]}
    parameters = {"similarity_threshold": 0.2, "ngram_range": (1, 2), "max_df": 0.1}
    pipe = Pipeline([
        ('processing',NLTKprocessing(**parameters))
    ])
    pipe.fit(company1,company2)
    pipe.predict(GroundTruth)'''

    '''parameters = {"similarity_threshold":[0.2],"ngram_range": [(1, 2)],"max_df": [0.1]}
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