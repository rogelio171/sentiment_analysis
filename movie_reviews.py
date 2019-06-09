import nltk

from nltk.sentiment import vader

def movie_reviews_sentiment_analysis():

    positive_reviews = open('rt-polarity.pos', 'r', encoding='ISO-8859-1').readlines()
    negative_reviews = open('rt-polarity.neg', 'r', encoding='ISO-8859-1').readlines()

    print(type(positive_reviews))

    print('The numbers of positive reviews are: {}'.format(len(positive_reviews)))
    print('The numbers of negative reviews are: {}'.format(len(negative_reviews)))

    vader_results = get_review_sentiments(positive_reviews, negative_reviews)

    run_analysis(vader_results)


def sentiment_calculator(sia, review):
    return sia.polarity_scores(review)['compound']

def get_review_sentiments(positive_reviews, negative_reviews):
    sia = vader.SentimentIntensityAnalyzer()
    positive_compound_scores = [sentiment_calculator(sia, positive_review) for positive_review in positive_reviews]
    negative_compound_scores = [sentiment_calculator(sia, negative_review) for negative_review in negative_reviews]
    return {'results-on-positive': positive_compound_scores, 'results-on-negative': negative_compound_scores}


def run_analysis(review_results):
    positive_review_results = review_results['results-on-positive']
    negative_review_results = review_results['results-on-negative']

    pct_True_Positive = float(sum(x > 0 for x in positive_review_results)) / len(positive_review_results)
    pct_True_Negative = float(sum(x < 0 for x in negative_review_results)) / len(negative_review_results)

    total_accurate = float(sum(x > 0 for x in positive_review_results)) + float(sum(x < 0 for x in negative_review_results))
    total = len(positive_review_results) + len(negative_review_results)

    print('Accuracy on positive reviews = {:.2f}%'.format(pct_True_Positive*100))
    print('Accuracy on negative reviews = {:.2f}%'.format(pct_True_Negative*100))
    print('Overall accuracy = {:.2f}%'.format(total_accurate*100/total))

if __name__ == "__main__":
    movie_reviews_sentiment_analysis()
    