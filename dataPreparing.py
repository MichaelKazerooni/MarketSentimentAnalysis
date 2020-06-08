import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import spacy
import json
import re
def generate_file_summary(data):
    file_sum = data.groupby('sentiment').nunique()
    print(file_sum)



def conv_json_news_headlines_to_pandas():
    def _prepare_sentiment(data):
        data.drop(columns=['company', 'id'], inplace=True)
        data.loc[((0.20 >= data['sentiment']) & (data['sentiment']>= -0.20)), 'sentiment'] = 0
        data.loc[-0.20 > data['sentiment'], 'sentiment'] = -1
        data.loc[data['sentiment'] > 0.20, 'sentiment'] = 1
        data.loc[data['sentiment'] == 0,'sentiment'] = '0'
        data.loc[data['sentiment'] == 1, 'sentiment'] = '1'
        data.loc[data['sentiment'] == -1, 'sentiment'] = '-1'
        return data

    paper_train_data = pd.read_json(r'C:\Users\Michael\PycharmProjects\MarketSentimentAnalysis\Headline_Trainingdata.json')
    paper_valid_data = pd.read_json(r'C:\Users\Michael\PycharmProjects\MarketSentimentAnalysis\Headline_Trialdata.json')
    paper_train_data = _prepare_sentiment(paper_train_data)
    paper_valid_data = _prepare_sentiment(paper_valid_data)

    return paper_train_data, paper_valid_data

def read_csv_file():
    def transform_sentiment():
        kaggle_news_headline.loc[kaggle_news_headline['sentiment']== 'neutral','sentiment'] = '0'
        kaggle_news_headline.loc[kaggle_news_headline['sentiment'] == 'positive', 'sentiment'] = '1'
        kaggle_news_headline.loc[kaggle_news_headline['sentiment'] == 'negative', 'sentiment'] = '-1'
    kaggle_news_headline = pd.read_csv(r'C:\Users\Michael\PycharmProjects\MarketSentimentAnalysis\sentiment analysis for financial news from retail investor view\all-data.csv', encoding = "ISO-8859-1", header = None)
    kaggle_news_headline.rename(columns = {0:'sentiment', 1: 'title'}, inplace= True)
    transform_sentiment()
    return kaggle_news_headline

def combine_data(kaggle, paper_train, paper_valid):
    final_data = pd.concat([kaggle, paper_train, paper_valid])
    final_data.drop_duplicates(keep = 'first', inplace = True)
    final_data.reset_index(inplace= True, drop= True)
    return final_data

def preprocess_data(data):
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    def _apply_regex(row):
        row = REPLACE_NO_SPACE.sub("",row)
        row = REPLACE_WITH_SPACE.sub(" ", row)
        return row
    data['title'] = data['title'].apply(lambda row: _apply_regex(row))
    # print(f'num is{round(len(data)*0.9)}')
    train_id = round(len(data)*0.9)
    train = data.iloc[0:train_id]
    test = data.iloc[train_id:]
    # reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in data]
    # reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    # return train, test
    return data
def logistic_regression(data):
    x_train, x_test, y_train, y_test = train_test_split(data['title'],data['sentiment'], random_state= 0)
    text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), lowercase= True)
    x_train_transformed = text_transformer.fit_transform(x_train)
    x_test_transformed = text_transformer.transform(x_test)
    logit = LogisticRegression(C=1, multi_class='multinomial', random_state=17)
    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=17)
    cv_results = cross_val_score(logit, x_train_transformed, y_train, cv = 10, scoring= 'accuracy')
    print(cv_results)
    print(cv_results.mean())

def multi_model_testing(data):
    text_transformer = TfidfVectorizer(stop_words= 'english', ngram_range=(1,2), lowercase= True)
    CV = 10
    result_list =[]
    tfidf_feature = text_transformer.fit_transform(data['title'])
    models = [
        RandomForestClassifier(n_estimators = 200, max_depth = 3, random_state = 0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state = 0),
    ]
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, tfidf_feature, data['sentiment'], scoring= 'accuracy', cv = CV)
        for fold_id, accuracy in enumerate(accuracies):
            result_list.append((model_name , fold_id, accuracy))
    cv_df = pd.DataFrame(result_list, columns=['model_name', 'fold_idx', 'accuracy'])
    print(cv_df.groupby('model_name')['accuracy'].mean())


# def multi_model():
#     def _evaluate_model():
#         pass
#
#     models = [
#         RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
#         LinearSVC(),
#         MultinomialNB(),
#         LogisticRegression(random_state=0),
#     ]
#     for model in models:
#         pipeline = Pipeline([
#             (ve)
#         ])

if __name__ == '__main__':
    kaggle_news_headline = read_csv_file()
    paper_train_headline, paper_valid_headline = conv_json_news_headlines_to_pandas()
    final_data = combine_data(kaggle_news_headline, paper_train_headline,paper_valid_headline)
    # print(len(final_data))
    generate_file_summary(final_data)
    data = preprocess_data(final_data)
    # logistic_regression(data)
    multi_model_testing(data)
    # print(len(train))
    # final_data.to_csv(r'C:\Users\Michael\PycharmProjects\MarketSentimentAnalysis\final_data.csv')
