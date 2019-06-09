import nltk

from nltk.sentiment import vader

if __name__ == "__main__":
    sia = vader.SentimentIntensityAnalyzer()
    phrase = 'What a terrible restaurant'
    polarity_scores = sia.polarity_scores(phrase)
    print('For the phrase:\n - {} \nVader generates the following results: {}'.format(phrase, polarity_scores))
    
    word = 'Terrible'
    polarity_scores = sia.polarity_scores(word)
    print('For the word:\n - {} \nVader generates the following results: {}'.format(word, polarity_scores))

    emoticon = ':)'
    polarity_scores = sia.polarity_scores(emoticon)
    print('For the emoticon:\n{}\nVader generates the following results: {}'.format(emoticon, polarity_scores))

    emoticon2 = ':/'
    polarity_scores = sia.polarity_scores(emoticon2)
    print('For the emoticon:\n{}\nVader generates the following results: {}'.format(emoticon2, polarity_scores))

    phrase2 = 'The cumin was the kiss of death'
    polarity_scores = sia.polarity_scores(phrase2)
    print('For the phrase:\n - {} \nVader generates the following results: {}'.format(phrase2, polarity_scores))

    phrase3 = 'The food was good'
    polarity_scores = sia.polarity_scores(phrase3)
    print('For the phrase:\n - {} \nVader generates the following results: {}'.format(phrase3, polarity_scores))
    
    phrase4 = 'The food was GOOD'
    polarity_scores = sia.polarity_scores(phrase4)
    print('For the phrase:\n - {} \nVader generates the following results: {}'.format(phrase4, polarity_scores))

    phrase5 = 'The food was good!'
    polarity_scores = sia.polarity_scores(phrase5)
    print('For the phrase:\n - {} \nVader generates the following results: {}'.format(phrase5, polarity_scores))

    phrase6 = 'The food was good!!!'
    polarity_scores = sia.polarity_scores(phrase6)
    print('For the phrase:\n - {} \nVader generates the following results: {}'.format(phrase6, polarity_scores))
    
    phrase7 = 'The food was not good!!!'
    polarity_scores = sia.polarity_scores(phrase7)
    print('For the phrase:\n - {} \nVader generates the following results: {}'.format(phrase7, polarity_scores))

    phrase8 = 'The food was not the worst!!!'
    polarity_scores = sia.polarity_scores(phrase8)
    print('For the phrase:\n - {} \nVader generates the following results: {}'.format(phrase8, polarity_scores))

    phrase9 = 'I usually hate seafood but I liked this'
    polarity_scores = sia.polarity_scores(phrase9)
    print('For the phrase:\n - {} \nVader generates the following results: {}'.format(phrase9, polarity_scores))
