import pandas as pd
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk import FreqDist
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

#read data from json filr
data = pd.read_json('politifact_factcheck_data.json', lines=True)

#classify multiclass into binary class
#data['verdict_binary'] = data['verdict'].map({'pants-fire' : 1.0,'false': 2.0, 'mostly-false': 3.0, 'half-true': 4.0, 'mostly-true': 5.0, 'true': 6.0})
data['verdict_binary'] = data['verdict'].map({'pants-fire' : 1.0,'false': 1.0, 'mostly-false': 1.0, 'half-true': 2.0, 'mostly-true': 2.0, 'true': 2.0})

#clean text to that no punctuation and all lowercase
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text
data['clean_text'] = data['statement'].apply(clean_text)

#reduce stopwords
def remove_stopwords(text):
    words = word_tokenize(text)
    clean_words = [word for word in words if word not in stopwords.words('english')]
    after_text = ' '.join(clean_words)
    return after_text
data['clean_text_no_stopword'] = data['clean_text'].apply(remove_stopwords)

#create statement, statement_source, statement_originator, and label set
feature = data.clean_text
label = data.verdict_binary
statement_source = data.statement_source
statement_originator = data.statement_originator

#method extract n-gram features
def extract_ngram(n, text):
    tokens = nltk.word_tokenize(text)
    bigrams = list(nltk.ngrams(tokens, n))
    return bigrams

#apply extracting ngram feature
feature_1grams = feature.apply(lambda x: extract_ngram(1, x))
feature_2grams = feature.apply(lambda x: extract_ngram(2, x))
feature_3grams = feature.apply(lambda x: extract_ngram(3, x))

# Select the top m most common bigrams for true and false without stopwords
def top_m_ngram_extractor(m, n):
    ngram_counts_by_value = {}
    for value in data['verdict_binary'].unique():
        subset_data = data[data['verdict_binary'] == value]
        ngrams = [ngram for text in subset_data['clean_text_no_stopword'] for ngram in extract_ngram(n, text)]
        ngram_freq_dist = FreqDist(ngrams)
        ngram_counts_by_value[value] = ngram_freq_dist

    top_m_ngrams_by_value = {}
    for value, freq_dist in ngram_counts_by_value.items():
        top_m_ngrams = freq_dist.most_common(m)
        top_m_ngrams_by_value[value] = [ngram for ngram, count in top_m_ngrams]
    return top_m_ngrams_by_value

#apply too m ngram feature dict{true:[],false:[]}
top_100_2grams = top_m_ngram_extractor(100, 2)
top_100_3grams = top_m_ngram_extractor(100, 3)

#ngrams without stopwords
no_stopwords_1grams = data.clean_text_no_stopword.apply(lambda x: extract_ngram(1, x))
no_stopwords_2grams = data.clean_text_no_stopword.apply(lambda x: extract_ngram(2, x))
no_stopwords_3grams = data.clean_text_no_stopword.apply(lambda x: extract_ngram(3, x))

def get_stopw_ngram(n):
    if n == 1:
        return no_stopwords_1grams
    elif n == 2:
        return no_stopwords_2grams
    elif n == 3:
        return no_stopwords_3grams

#methof to combine different feature
def combine_features(f1, f2, n):
    combined_features = []
    for i in range(len(data)):
        top_ngram = set(f2[label.iloc[i]])
        n_gram_i = set(get_stopw_ngram(n).iloc[i]).intersection(top_ngram)
        combined_feature = list(f1.iloc[i]) + list(n_gram_i)
        combined_features.append(combined_feature)
    return combined_features

#method conbine features and extra feature
def plug_in_extra(f1):
    combined_features = []
    for i in range(len(data)):
        source_tuple = (data.statement_source.iloc[i],)
        originator_tuple = (data.statement_originator.iloc[i],)
        combined_feature = list(f1[i])
        combined_feature.append(originator_tuple)
        combined_feature.append(source_tuple)
        combined_features.append(combined_feature)
    return combined_features

#first grid search
# Perform manual grid search
best_accuracy = 0.0
best_params = {}

# Define the feature options, model options, and hyperparameter options
feature_options = ['unigrams_no_stopwords', 'unigrams', 'bigrams', 'trigram']
model_options = ['LogisticRegression', 'MultinomialNB', 'KNeighborsClassifier']
hyperparameter_options = {
    'LogisticRegression': [1.0, 0.5, 2.0],
    'MultinomialNB': [0.1, 0.2, 0.5],
    'KNeighborsClassifier': [3, 5, 7]
}

# Iterate through all combinations
for feature_option in feature_options:
    for model_option in model_options:
        for hyperparameter_option in hyperparameter_options[model_option]:
            params = {
                'features': feature_option,
                'model': model_option,
                'hyperparameters': hyperparameter_option
            }
            print(f"Testing Parameters: {params}, ", end='')

            # Extract features based on the chosen configuration
            if feature_option == 'unigrams':
                cf = feature_1grams
            elif feature_option == 'unigrams_no_stopwords':
                cf = no_stopwords_1grams
            elif feature_option == 'bigrams':
                cf = feature_2grams
            elif feature_option == 'trigram':
                cf = feature_3grams

            # Split the data
            label_train, label_test, feature_train, feature_test = train_test_split(label, cf, test_size=0.1, random_state=114)
            label_train, label_dev, feature_train, feature_dev = train_test_split(label_train, feature_train, test_size=0.125, random_state=114)

            # Convert features into a sparse matrix
            dicttrain = [{feature: label_train.iloc[i] for feature in features} for i, features in enumerate(feature_train)]
            dictdev = [{feature: label_dev.iloc[i] for feature in features} for i, features in enumerate(feature_dev)]
            vectorizer = DictVectorizer()
            feature_train_vectorized = vectorizer.fit_transform(dicttrain)
            feature_dev_vectorized = vectorizer.transform(dictdev)

            # Initialize and train the model
            if model_option == 'LogisticRegression':
                model = LogisticRegression(C=params['hyperparameters'])
            elif model_option == 'MultinomialNB':
                model = MultinomialNB(alpha=params['hyperparameters'])
            elif model_option == 'KNeighborsClassifier':
                model = KNeighborsClassifier(n_neighbors=params['hyperparameters'])

            model.fit(feature_train_vectorized, label_train)

            # Make predictions on the development set
            prediction = model.predict(feature_dev_vectorized)

            # Calculate accuracy
            accuracy = accuracy_score(label_dev, prediction)
            print(f'Accuracy: {round(accuracy * 100, 2)}')

            # Update the best parameters if the current configuration is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params

# Print the best parameters
print("\nBest Parameters:")
print(best_params)
print(f"Best Accuracy: {round(best_accuracy * 100, 2)}")


#second grid search
# Perform manual grid search
best_accuracy = 0.0
best_params = {}

# Define the feature options, model options, and hyperparameter options
feature_options = ['usw_100_bigram', 'u_100_bigram', 'usw_100_trigram', 'u_100_trigram']
model_options = ['LogisticRegression', 'MultinomialNB']

# Iterate through all combinations
for feature_option in feature_options:
    for model_option in model_options:
        params = {
            'features': feature_option,
            'model': model_option,
        }
        print(f"Testing Parameters: {params}, ", end='')

        # Extract features based on the chosen configuration
        if feature_option == 'usw_100_bigram':
                cf = combine_features(no_stopwords_1grams, top_100_2grams, 2)
        elif feature_option == 'u_100_bigram':
                cf = combine_features(feature_1grams, top_100_2grams, 2)
        elif feature_option == 'usw_100_trigram':
            cf = combine_features(no_stopwords_1grams, top_100_3grams, 3)
        elif feature_option == 'u_100_trigram':
            cf = combine_features(feature_1grams, top_100_3grams, 3)

        # Split the data
        label_train, label_test, feature_train, feature_test = train_test_split(label, cf, test_size=0.1, random_state=114)
        label_train, label_dev, feature_train, feature_dev = train_test_split(label_train, feature_train, test_size=0.125, random_state=114)

        # Convert features into a sparse matrix
        dicttrain = [{feature: label_train.iloc[i] for feature in features} for i, features in enumerate(feature_train)]
        dictdev = [{feature: label_dev.iloc[i] for feature in features} for i, features in enumerate(feature_dev)]
        vectorizer = DictVectorizer()
        feature_train_vectorized = vectorizer.fit_transform(dicttrain)
        feature_dev_vectorized = vectorizer.transform(dictdev)

        # Initialize and train the model
        if model_option == 'LogisticRegression':
            model = LogisticRegression()
        elif model_option == 'MultinomialNB':
            model = MultinomialNB()

        model.fit(feature_train_vectorized, label_train)

        # Make predictions on the development set
        prediction = model.predict(feature_dev_vectorized)

        # Calculate accuracy
        accuracy = accuracy_score(label_dev, prediction)
        print(f'Accuracy: {round(accuracy * 100, 2)}')

        # Update the best parameters if the current configuration is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

# Print the best parameters
print("\nBest Parameters:")
print(best_params)
print(f"Best Accuracy: {round(best_accuracy * 100, 2)}")


#Fianl grid search
# Perform manual grid search
best_accuracy = 0.0
best_params = {}

# Define the feature options, model options, and hyperparameter options
feature_options = ['u_100_bigram', 'u_100_bigram_plus_extra']
hyperparameter_options = [0.1, 1.0, 1.25, 1.5]

# Iterate through all combinations
for feature_option in feature_options:
    for hyperparameter_option in hyperparameter_options:
        params = {
            'features': feature_option,
            'hyperparameters': hyperparameter_option,
        }
        print(f"Testing Parameters: {params}, ", end='')

        # Extract features based on the chosen configuration
        if feature_option == 'u_100_bigram':
            cf = combine_features(feature_1grams, top_100_2grams, 2)
        elif feature_option == 'u_100_bigram_plus_extra':
            k = combine_features(feature_1grams, top_100_2grams, 2)
            cf = plug_in_extra(k)

        # Split the data
        label_train, label_test, feature_train, feature_test = train_test_split(label, cf, test_size=0.1, random_state=114)
        label_train, label_dev, feature_train, feature_dev = train_test_split(label_train, feature_train, test_size=0.125, random_state=114)

        # Convert features into a sparse matrix
        dicttrain = [{feature: label_train.iloc[i] for feature in features} for i, features in enumerate(feature_train)]
        dictdev = [{feature: label_dev.iloc[i] for feature in features} for i, features in enumerate(feature_dev)]
        vectorizer = DictVectorizer()
        feature_train_vectorized = vectorizer.fit_transform(dicttrain)
        feature_dev_vectorized = vectorizer.transform(dictdev)

        # Initialize and train the model
        model = LogisticRegression(C=params['hyperparameters'])
        model.fit(feature_train_vectorized, label_train)

        # Make predictions on the development set
        prediction = model.predict(feature_dev_vectorized)

        # Calculate accuracy
        accuracy = accuracy_score(label_dev, prediction)
        print(f'Accuracy: {round(accuracy * 100, 2)}')

        # Update the best parameters if the current configuration is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

# Print the best parameters
print("\nBest Parameters:")
print(best_params)
print(f"Best Accuracy: {round(best_accuracy * 100, 2)}")


#test_set
# Define the features for testing
best_accuracy = 0.0
best_params = {}

# Define the feature options, model options, and hyperparameter options
feature_options = ['u_100_bigram', 'u_100_bigram_plus_extra']
hyperparameter_options = [1.0, 1.25]

# Iterate through all combinations
for feature_option in feature_options:
    for hyperparameter_option in hyperparameter_options:
        params = {
            'features': feature_option,
            'hyperparameters': hyperparameter_option,
        }
        print(f"Testing Parameters: {params}, ", end='')

        # Extract features based on the chosen configuration
        if feature_option == 'u_100_bigram':
            cf = combine_features(feature_1grams, top_100_2grams, 2)
        elif feature_option == 'u_100_bigram_plus_extra':
            k = combine_features(feature_1grams, top_100_2grams, 2)
            cf = plug_in_extra(k)

        # Split the data
        label_train, label_test, feature_train, feature_test = train_test_split(label, cf, test_size=0.1, random_state=114)
        label_train, label_dev, feature_train, feature_dev = train_test_split(label_train, feature_train, test_size=0.125, random_state=114)

        # Convert features into a sparse matrix
        dicttrain = [{feature: label_train.iloc[i] for feature in features} for i, features in enumerate(feature_train)]
        dicttest = [{feature: label_test.iloc[i] for feature in features} for i, features in enumerate(feature_test)]
        vectorizer = DictVectorizer()
        feature_train_vectorized = vectorizer.fit_transform(dicttrain)
        feature_test_vectorized = vectorizer.transform(dicttest)

        # Initialize and train the model
        model = LogisticRegression(C=params['hyperparameters'])
        model.fit(feature_train_vectorized, label_train)

        # Make predictions on the test set
        prediction = model.predict(feature_test_vectorized)

        # Calculate accuracy
        accuracy = accuracy_score(label_test, prediction)
        print(f'Accuracy: {round(accuracy * 100, 2)}')

        # Update the best parameters if the current configuration is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

# Print the best parameters
print("\nBest Parameters:")
print(best_params)
print(f"Best Accuracy: {round(best_accuracy * 100, 2)}")

#Best test_set configuration detail
k = combine_features(feature_1grams, top_100_2grams, 2)
cf = plug_in_extra(k)

# Split the data
label_train, label_test, feature_train, feature_test = train_test_split(label, cf, test_size=0.1, random_state=114)
label_train, label_dev, feature_train, feature_dev = train_test_split(label_train, feature_train, test_size=0.125, random_state=114)

# Convert features into a sparse matrix
dicttrain = [{feature: label_train.iloc[i] for feature in features} for i, features in enumerate(feature_train)]
dicttest = [{feature: label_test.iloc[i] for feature in features} for i, features in enumerate(feature_test)]
vectorizer = DictVectorizer()
feature_train_vectorized = vectorizer.fit_transform(dicttrain)
feature_test_vectorized = vectorizer.transform(dicttest)

# Initialize and train the model
model = LogisticRegression(C=1.25)
model.fit(feature_train_vectorized, label_train)

# Make predictions on the test set
prediction = model.predict(feature_test_vectorized)

# Calculate accuracy
accuracy = accuracy_score(label_test, prediction)
print(f'Accuracy: {round(accuracy * 100, 2)}')
print(classification_report(label_test, prediction))








