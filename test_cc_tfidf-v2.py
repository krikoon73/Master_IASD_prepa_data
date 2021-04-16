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

    def __init__(self, similarity_threshold, matches, number_of_matches,
                 big_manufacturer_list,max_ngram):
        logging.debug(">>>>>>>>NLTKProcessing->init() called.")
        self.similarity_threshold = similarity_threshold
        #self.max_df = max_df
        self.matches = matches
        self.number_of_matches = number_of_matches
        self.matches_df = pd.DataFrame(matches)
        self.big_manufacturer_list = big_manufacturer_list
        self.max_ngram = max_ngram

        logging.debug(f"similarity_threshold={self.similarity_threshold}")
        logging.debug(f"max_ngram=(1,{self.max_ngram}")
        #logging.debug(f"max_df={self.max_df}")
        logging.debug(f"len(matches)={len(self.matches)}")
        logging.debug(f"number_of_matches={number_of_matches}")
        logging.debug(f"big_manufacturer_list={big_manufacturer_list}")

    def TFIDF(self,X,filtre,max_ngram,max_df):
        logging.debug(">>>>>>>>NLTKProcessing->TFIDF() called.")
        logging.debug(f"max_ngram={max_ngram}")
        logging.debug(f"max_df={max_df}")
        vectorizer = TfidfVectorizer(ngram_range=(1,max_ngram), max_df=max_df,sublinear_tf=True,stop_words=[filtre])
        vectors = vectorizer.fit_transform(X['full data'])
        feature_names = vectorizer.get_feature_names()
        return vectors.todense()

    def prep(self,texte):
        # suppression des caracteres non alphanumériques + tout en minuscule
        texte = re.sub("[^a-zA-Z0-9_]", " ", str(texte)).lower()
        # remplacement de mots
        texte = texte.replace("professional", "pro").replace("windows","win").replace("upgrade","upg").replace("deluxe","dlx")
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

    def retreatprice(self,texte):
        # suppression des caracteres non alphanumériques + tout en minuscule
        return float(re.sub("[^0-9.]", " ", str(texte)))

    def generate_corpus(self,company1,company2):
        logging.debug(">>>>>>>>NLTKProcessing->generate_corpu() called.")

        company1['Company'] = "company1"
        company1 = company1.rename(columns={"title": "name"})
        company1['name'] = company1['name'].fillna(' ')
        company1['manufacturer'] = company1['manufacturer'].fillna(' ')
        company1['description'] = company1['description'].fillna(' ')
        company1['price'] = company1['price'].fillna(' ')
        company1['price_retreat'] = company1['price'].apply(self.retreatprice)
        company1['full data'] = company1['manufacturer'].apply(self.prep) + ' ' + company1['name'].apply(
            self.prep)  # + ' ' + company1['description'].apply(prep)

        company2['Company'] = "company2"
        company2['name'] = company2['name'].fillna(' ')
        company2['manufacturer'] = company2['manufacturer'].fillna(' ')
        company2['description'] = company2['description'].fillna(' ')
        company2['price'] = company2['price'].fillna(' ')
        company2['price_retreat'] = company2['price'].apply(self.retreatprice)
        company2['full data'] = company2['manufacturer'].apply(self.prep) + ' ' + company2['name'].apply(
            self.prep)  # + ' ' + company2['description'].apply(prep)

        #----

        company1['Win'] = np.where(company1['name'].str.contains('win') | company1['description'].str.contains('win'),1, 0)
        company1['Mac'] = np.where(company1['name'].str.contains('mac') | company1['description'].str.contains('mac'),1, 0)
        company1['Linux'] = np.where(company1['name'].str.contains('linux') | company1['description'].str.contains('linux'), 1, 0)
        company1['version'] = np.select(
            [(company1['Win'] + company1['Mac'] + company1['Linux'] == 1) & (company1['Win'] == 1),
             (company1['Win'] + company1['Mac'] + company1['Linux'] == 1) & (company1['Mac'] == 1),
             (company1['Win'] + company1['Mac'] + company1['Linux'] == 1) & (company1['Linux'] == 1),
             (company1['Win'] + company1['Mac'] + company1['Linux'] != 1)
             ],
            ['Win', 'Mac', 'Linux', 'None'])

        company2['Win'] = np.where(company2['name'].str.contains('win') | company2['description'].str.contains('win'),1, 0)
        company2['Mac'] = np.where(company2['name'].str.contains('mac') | company2['description'].str.contains('mac'),1, 0)
        company2['Linux'] = np.where(company2['name'].str.contains('linux') | company2['description'].str.contains('linux'), 1, 0)
        company2['version'] = np.select(
            [(company2['Win'] + company2['Mac'] + company2['Linux'] == 1) & (company2['Win'] == 1),
             (company2['Win'] + company2['Mac'] + company2['Linux'] == 1) & (company2['Mac'] == 1),
             (company2['Win'] + company2['Mac'] + company2['Linux'] == 1) & (company2['Linux'] == 1),
             (company2['Win'] + company2['Mac'] + company2['Linux'] != 1)
             ],
            ['Win', 'Mac', 'Linux', 'None'])
        #----

        corpus = pd.concat([company1, company2], sort=False, ignore_index=True)
        logging.debug(f"Corpus len {len(corpus)}")
        return company1,company2,corpus

    def run(self,company1,company2,corpus,filtre,similarity_threshold,max_ngram,max_df):
        new_number_of_matches = 0
        new_matches = []

        dense = self.TFIDF(corpus,filtre,max_ngram,max_df)
        self.dense = dense

        n1 = len(company1)
        n2 = len(company2)

        #for i in tqdm(range(n1), position=0, desc="Loop on company1", leave=False, colour='green', ncols=80):
        for i in tqdm(range(n1),position=0,desc="Outer loop",leave=False,colour="green",ncols=80):
            price1 = float(company1.iloc[i, 6])
            #for j in tqdm(range(n2), position=1, desc="Loop on company2", leave=False, colour='red', ncols=80):
            for j in tqdm(range(n2),position=1,desc="Inner loop",leave=False,colour="blue",ncols=80):
                # try :
                price2 = float(company2.iloc[j, 6])
                if (price1 * price2 == 0 or (max(price1, price2) / min(price1, price2) < 2)) and \
                        (company1.iloc[i, 11] == company2.iloc[j, 11] or company1.iloc[i, 11] == 'None' or company2.iloc[j, 11] == 'None'):
                    try:
                        similarity = np.dot(dense[i], np.transpose(dense[len(company1) + j])).item(0) / math.sqrt(
                            np.dot(dense[i], np.transpose(dense[i])).item(0) * np.dot(dense[n1 + j],np.transpose(dense[n1 + j])).item(0))
                    except:
                        similarity = 0
                    if similarity > similarity_threshold:  # or jd_ng1_ng2_name<0.2 :# or name_score<=1) :
                        new_number_of_matches = new_number_of_matches + 1
                        new_matches.append((company1.iloc[i, 0], company2.iloc[j, 0]))

        if new_number_of_matches > 0:
            logging.debug("New matches: {}".format(new_number_of_matches))
            self.number_of_matches = self.number_of_matches + new_number_of_matches
            logging.debug("Total matches: {}".format(self.number_of_matches))
            if self.matches == []:
                self.matches = new_matches
                self.matches_df = pd.DataFrame(self.matches)
                self.matches_df.columns = ['idCompany1', 'idCompany2']
            else:
                new_matches_df = pd.DataFrame(new_matches)
                new_matches_df.columns = ['idCompany1', 'idCompany2']
                self.matches_df = pd.concat([self.matches_df, new_matches_df], sort=False,
                                            ignore_index=True).drop_duplicates()
            logging.info(f"New matches : {new_number_of_matches}")
            logging.info(f"Total matches : {self.number_of_matches}")

    def fit(self, company1,company2):
        logging.debug(">>>>>>>>NLTKProcessing->fit() called.")
        logging.debug(f"len(company1)={len(company1)}")
        logging.debug(f"len(company2)={len(company2)}")

        results = self.generate_corpus(company1,company2)
        company1 = results[0]
        company2 = results[1]
        corpus = results[2]

        #number_of_matches = 0

        company1_light = company1
        company2_light = company2

        logging.info("Start search dedup STEP1")
        for filtre in self.big_manufacturer_list:
            logging.info(f"Start search depup for big manufacturer {filtre}")
            for max_ngram in [2, 1]:
                logging.info(f"ngram_range=(1,{max_ngram})")
                try:
                    company1_light = company1[~company1.id.isin(self.matches_df.idCompany1)]
                    company2_light = company2[~company2.id.isin(self.matches_df.idCompany2)]
                except:
                    company1_light = company1
                    company2_light = company2
                company1_light = company1_light[company1_light['full data'].str.contains(filtre)].reset_index(drop=True)
                company2_light = company2_light[company2_light['full data'].str.contains(filtre)].reset_index(drop=True)
                corpus = pd.concat([company1_light, company2_light], sort=False, ignore_index=True)
                logging.debug("{} with ngram=(1,{})".format(filtre, max_ngram))
                self.run(company1_light,company2_light,corpus,filtre,self.similarity_threshold,max_ngram,0.99)
                # filtre_tfidf(corpus, (1, max_ngram), 0.1, 0.6, filtre)
            logging.debug(f"Total matchs {len(self.matches_df)}")
            logging.debug(f"End filtering for big manufacturer {filtre}")
        logging.info("End search dedup STEP1")
        logging.info("Start search dedup STEP2")
        for max_ngram in [3, 2, 1]:
            logging.info(f"ngram_range=(1,{max_ngram})")
            company1_light = company1[~company1.id.isin(self.matches_df.idCompany1)]
            company2_light = company2[~company2.id.isin(self.matches_df.idCompany2)]
            corpus = pd.concat([company1_light, company2_light], sort=False, ignore_index=True)
            logging.debug("Other entries with ngram=(1,{})".format(max_ngram))
            self.run(company1_light, company2_light, corpus, "", (self.similarity_threshold)/1.2,max_ngram,0.01)
        logging.info("End search dedup STEP2")

    def predict(self,ground_truth_matches):
        logging.debug(">>>>>>>>NLTKprocessing->predict() called.")
        logging.info("Compute scoring for epoch")
        #matches_df = pd.DataFrame(self.matches)
        #matches_df = self.matches_df
        #print(self.matches_df)
        self.matches_df.columns = ['idCompany1', 'idCompany2']
        diff_df = pd.merge(ground_truth_matches, self.matches_df, how='outer', indicator='Exist')
        true_positives = diff_df[diff_df.Exist == 'both']
        false_positives = diff_df[diff_df.Exist == 'right_only']
        false_negatives = diff_df[diff_df.Exist == 'left_only']
        logging.info("Number of true positives: {}".format(len(true_positives)))
        logging.info("Number of false positives: {}".format(len(false_positives)))
        logging.info("Number of false negatives: {}".format(len(false_negatives)))
        precision = len(true_positives) / (len(true_positives) + len(false_positives))
        logging.info("Precision: {}".format(precision))
        recall = len(true_positives) / (len(true_positives) + len(false_negatives))
        logging.info("Recall: {}".format(recall))
        f_measure = 2 * (precision * recall) / (precision + recall)
        logging.info("F measure: {}".format(f_measure))
        return true_positives,false_positives,false_negatives,precision,recall,f_measure,diff_df

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
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
    stop_words.update(["r", "v", "software", "entertainment", "inc", "usa"])

    df_load = loadData(pathFileCompany1,pathFileCompany2,pathFileGroundTruth)
    company1 = df_load[0]
    company2 = df_load[1]
    GroundTruth = df_load[2]

    score = pd.DataFrame(columns=['nb_ref_company1',
                                  'nb_ref_company2',
                                  #'ngram_range',
                                  #'max_df',
                                  'similarity_threshold',
                                  'true_positives',
                                  'false_positives',
                                  'false_negatives',
                                  'precision',
                                  'recall',
                                  'f_measure',
                                  'processing_time'])
    max_ngram = 3

    Min = 0.
    Max = 1
    step = 0.1
    s = step
    epoch = 0
    epochs = round((Max-Min)/step)
    big_manufacturer_list=["adobe","encore","topic","microsoft","aspyr","apple","fogware","intuit","punch","sony","nova","corel"]
    t_start = time()
    for s in [0.1,0.2,0.3,0.6,0.9]:
        epoch += 1
        logging.info(f"epoch={epoch} - similiraty_threshold={round(s, 2)} - START")
        t_start_model = time()
        number_of_matches = 0
        matches = []
        matches_df = []
        model = NLTKprocessing(s,matches,number_of_matches, big_manufacturer_list,max_ngram)
        Y_pred = model.fit(company1, company2)
        results = model.predict(GroundTruth)
        t_stop_model = time()
        true_positives = results[0]
        false_positives = results[1]
        false_negatives = results[2]
        precision = results[3]
        recall = results[4]
        f_measure = results[5]
        score = score.append({'nb_ref_company1': len(company1),
                              'nb_ref_company2': len(company2),
                              #'ngram_range': ngram_range,
                              #'max_df': max_df,
                              'similarity_threshold': s,
                              'true_positives': len(true_positives),
                              'false_positives': len(false_positives),
                              'false_negatives': len(false_negatives),
                              'precision': precision,
                              'recall': recall,
                              'f_measure': f_measure,
                              'processing_time': t_stop_model - t_start_model},
                             ignore_index=True)
        logging.info(
            f"epoch={epoch} - similiraty_threshold={round(s, 2)} - time = {t_stop_model - t_start_model} / f_measure={f_measure}")
        #s += step

    t_stop = time()
    logging.info(f"PROCESSING_TIME={t_stop - t_start}")

    score['tpr'] = score['true_positives'] / (score['true_positives'] + score['false_negatives'])
    nb_distinct_ref_company1 = len(pd.unique(company1['id']))
    logging.debug(f"nb_distinct_ref_company1={nb_distinct_ref_company1}")
    nb_distinct_ref_company2 = len(pd.unique(company2['id']))
    logging.debug(f"nb_distinct_ref_company2={nb_distinct_ref_company2}")
    # score['true_negatives'] = abs(score['true_positives']-min(nb_distinct_ref_company1,nb_distinct_ref_company2))
    score['true_negatives'] = nb_distinct_ref_company1 * nb_distinct_ref_company2 - score['true_positives'] - score[
        'false_positives'] - score['false_negatives']

    score['fpr'] = score['false_positives'] / (score['true_negatives'] + score['false_positives'])
    filename_score = "results/scores_run-full.csv"
    score.to_csv(filename_score, index=False)


