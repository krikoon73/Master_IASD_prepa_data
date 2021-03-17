import pandas as pd
import nltk
from tqdm import tqdm
from matplotlib import pyplot as plt
import re

import time

class NLTKPreprocessor1():
    def __init__(self):
        print(">>>>>>>>NLTKPreprocessor1->init() called.")
        '''
        self.df1 = pd.DataFrame()
        self.df2 = pd.DataFrame()
        self.jd = jd
        self.ns = ns
        self.x = pd.DataFrame()
        '''

    def fit(self,df1,df2,**fit_params):
        print(">>>>>>>>NLTKPreprocessor1->fit() called.")
        print(f"len(df1)={len(df1)}")
        print(f"len(df2)={len(df2)}")
        print(f"jd = {jd}")
        print(f"ns = {ns}")
        return self

    def TOKENIZE(self,x,index):
        #print("\n>>>>>>>>TOKENIZE() called.\n")
        # Delete punctuation
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        return tokenizer.tokenize(x.iloc[index,1])

    def transform(self,df1,df2,**fit_params):
        print(">>>>>>>>NLTKPreprocessor1->transform() called.")
        matches = []
        number_of_matches = 0
        #df_score = pd.DataFrame()
        for i in tqdm(range(len(df1)),desc="modeling"):
            #tokens1name = tokenizer.tokenize(df1.iloc[i, 1])
            tokens1name = self.TOKENIZE(df1,i)
            ng1_tokensname = set(nltk.ngrams(tokens1name, n=1))
            for j in range(len(df2)):
                #tokens2name = tokenizer.tokenize(df2.iloc[j, 1])
                tokens2name = self.TOKENIZE(df2,j)
                ng2_tokensname = set(nltk.ngrams(tokens2name, n=1))
                jd_ng1_ng2_name = nltk.jaccard_distance(ng1_tokensname, ng2_tokensname)
                name_score = nltk.edit_distance(df1.iloc[i, 1], df2.iloc[j, 1])
                if (jd_ng1_ng2_name <= jd) or (name_score <= ns):
                    number_of_matches = number_of_matches + 1
                    matches.append((df1.iloc[i, 0], df2.iloc[j, 0]))
        print("Number of matches: {}".format(number_of_matches))
        return number_of_matches,matches

class NLTKPreprocessor2():
    def __init__(self):
        print(">>>>>>>>NLTKPreprocessor2->init() called.")

    def fit(self,df1,df2,**fit_params):
        print(">>>>>>>>NLTKPreprocessor2->fit() called.")
        print(f"len(df1)={len(df1)}")
        print(f"len(df2)={len(df2)}")
        print(f"jd = {jd}")
        print(f"ns = {ns}")
        return self

    def prep(texte):
        # suppression des caracteres non alphanumériques + tout en minuscule
        texte = re.sub("[^a-zA-Z0-9_]", " ", str(texte)).lower()
        # tokenization par mot
        tokens = nltk.word_tokenize(texte)
        # supression des stopwords
        filtered_tokens = [w for w in tokens if not w in stop_words]
        # Stemming
        texte = [nltk.stem.SnowballStemmer('english').stem(w) for w in filtered_tokens]
        # remise sous forme d'une string
        return " ".join(texte)

class dedup_report:
    def __init__(self):
        print(">>>>>>>>depu_report->init() called.")
        self.df_GroundTruth = pd.DataFrame()
        #self.df_matches = pd.DataFrame()
        self.matches = []

    def processing(self,df_GroundTruth,matches):
        print(">>>>>>>>dedup_report->processing() called.")
        matches_df = pd.DataFrame(matches)
        matches_df.columns = ['idCompany1', 'idCompany2']
        diff_df = pd.merge(df_GroundTruth, matches_df, how='outer', indicator="Exist")
        true_positives = diff_df[diff_df.Exist == 'both']
        false_positives = diff_df[diff_df.Exist == 'right_only']
        false_negatives = diff_df[diff_df.Exist == 'left_only']
        precision = len(true_positives) / (len(true_positives) + len(false_positives))
        recall = len(true_positives) / (len(true_positives) + len(false_negatives))
        f_measure = 2 * (precision * recall) / (precision + recall)
        print(f"precision : {precision}")
        print(f"recall : {recall}")
        print(f"f_measure : {f_measure}\n")
        return precision,recall,f_measure,diff_df

if __name__ == '__main__':
    df_SampleCompany1 = pd.read_csv("SampleData/Sample_Company1.csv")
    df_SampleCompany2 = pd.read_csv("SampleData/Sample_Company2.csv")
    df_GroundTruth = pd.read_csv("Data/Ground_truth_mappings.csv")
    stop_words = set(nltk.corpus.stopwords.words('english'))

    col_names = ['ns', 'jd', 'precision', 'recall', 'f_measure']
    final_results = pd.DataFrame(columns=col_names)

    fit_params = {"jd":[0.7,0.71,0.72,0.73,0.74,0.75],"ns":[1]}
    print(fit_params)
    TEST1 = NLTKPreprocessor1()
    REPORT1 = dedup_report()
    print("\n")
    for ns in fit_params['ns']:
        for jd in fit_params['jd']:
            print(f">>>>>>>> ns={ns} jd={jd}")
            result1 = TEST1.transform(df_SampleCompany1,df_SampleCompany2,**fit_params)
            matches = result1[1]
            R1 = REPORT1.processing(df_GroundTruth,matches)
            new_row = {'ns':ns,'jd':jd, 'precision':R1[0],'recall':R1[1],'f_measure':R1[2]}
            final_results = final_results.append(new_row,ignore_index=True)
    final_results.plot(x='jd',y=['precision','recall','f_measure'])
    plt.show()








