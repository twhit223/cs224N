import numpy as np
import pandas as pd
import nltk
import random
import pickle
import code
import string
import os

from statistics import mode
from time import localtime, strftime
from collections import Counter
from scipy.stats import ttest_ind, ttest_rel

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, LassoCV
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

print(strftime("%Y-%m-%d %H:%M:%S", localtime()))

# !!! This code is a modified version of code written by Tyler Whittle that analyzes both the difference in scores from Glassdoor reviews


# Set the parameters
num_features = 2000
print("num_features: " + str(num_features))
# Indicates whether to only use data from the software & internet industries
swint = False
print("swint: " + ("True" if swint else "False"))
# Indicates whether to use pre-IPO and post-IPO: "both", "before", "after". Must be set to both for "pre_post" analysis.
before_after = "both"
# Word process can be either: "raw","stem", "lemma", or "features"
word_process = "raw"
# Classification scheme: "score" or "pre_post"
classification_unit = "score"
print("classification_unit: " + classification_unit)
# GBN indicates whether reviews include Good/Bad/Neutral (True) or just Good/Bad (False)
gbn = True
# Subset indicates whether to look at the good/bad/neutral subset. Can be either "good", "neutral", "bad", or "all"
gbn_subset = "all"
print("gbn_subset: " + gbn_subset)
# Thresh indicate the threshold that good/bad companies are cut off at. This should be entered as a string in the form of "0D". For instance, if the threshold is 0.25, it would be represented as "025". For a threshold of 0.5, it would be "05". 
gbn_thresh = "025"
# Score split can be either "binary","trinary", or "stars". Binary splits it into good/bad with good being 4,5 stars and bad being 1,2 stars. This removes the 3 star reviews. Trinary splits it into good/neutral/bad with 3 star reviews being neutral. 
score_split = "stars"
print("score_split: " + score_split)
# Labels indicates whether or not to append _pro, _con, _adv to the components of the reviews
labels = False
# N-gram can be either 1 for single words or 2 for bigrams
n_gram = 1
print("n_gram: " + str(n_gram))
# Years after IPO can be either 1 or 2
years_after_ipo = 2
print("years_after_ipo: " + str(years_after_ipo))
# Sampled indicates whether a reivew was sampled to be trained/tested in the classifier
sampled = 0

# Features - all are stemmed words based on the PorterStemmer()
compensation = set(("compens", "pay", "paid", "benefit", "perk", "401k", "retir"))
product = set(("product", "technolog", "softwar"))
career_dev = set(("promot", "opportun", "career"))
culture = {"cultur"}
wl_bal = {"worklife"}
politics = set(("politic", "nepostism", "honest", "ethic"))
boys_club = set(("boy", "frat", "sex"))
manager = set(("manag", "supervisor", "director"))
startup = {"startup"}
leader = set(("leader", "exec", "execut", "CEO", "CTO", "COO"))
strategy = set(("strategi", "vision", "mission"))
sales = {"sale"}
communication = set(("commun", "transpar"))
benevolence = set(("valu", "integr", "trust", "respect", "welfar", "care"))
finance = set(("cost", "financ", "wallst", "audit", "stock"))


# all_features = benevolence
all_features = compensation|product|career_dev|culture|wl_bal|politics|boys_club|manager|startup|leader|strategy|sales|communication|benevolence|finance


# Get the data
print("Fetching files...")
revs_before = []
if before_after == "both" or before_after == "before":
  # revs_before = open(DATA_DIR + "/reviews_text/reviews_before" + ("_swint" if swint else "_all") + ("_gbn" if gbn else "") + ("_1yr" if years_after_ipo is 1 else "") + "_thresh_" + gbn_thresh + ".txt", "r").read().split('\n')
  revs_before = open('data/reviews_all_public.txt', 'r').read().split('\n')
  revs_before = revs_before[0:100000]
  print("Fetched " + str(len(revs_before)) + " before reviews...")

revs_after = []
# if before_after == "both" or before_after == "after":
#   revs_after = open(DATA_DIR + "/reviews_text/reviews_after" + ("_swint" if swint else "_all") + ("_gbn" if gbn else "") + ("_1yr" if years_after_ipo is 1 else "") + "_thresh_" + gbn_thresh + ".txt", "r").read().split('\n')
#   print("Fetched " + str(len(revs_after)) + " after reviews...")

revs_raw = revs_after + revs_before


print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
print("Preprocessing reviews..")

# Pre-process the reviews by removing stopwords, stemming/lemmatizing, and appending the category to the words
def preprocess_reviews(revs, word_process):
  try:
    stop_words = set(stopwords.words("english"))
    processed_revs = []
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    print(len(revs))

    for rev in revs:
      # Get the categories
      if rev == '':
        continue
      rev = rev.split('|~|')
      pro = rev[0]
      con = rev[1]
      adv = rev[2]
      score = rev[3]
      pre_post = rev[4]
      rev_id = rev[5]
      company_id = rev[6]
      industry = rev[7]
      comp_good_bad = rev[8]

      # Tokenize the words
      pro_words = word_tokenize(pro)
      con_words = word_tokenize(con)
      adv_words = word_tokenize(adv)
      
      if word_process == "raw":
        # Option 1: Filter the stopwords out and append the category
        filtered_pros = [w.lower() for w in pro_words if not w in stop_words]
        filtered_cons = [w.lower()  for w in con_words if not w in stop_words]
        filtered_adv = [w.lower()  for w in adv_words if not w in stop_words]
      elif word_process == "stem":
        # Option 2: Filter the stopwords out, stem the words, and append the category 
        filtered_pros = [ps.stem(w).lower() + ("_pro" if labels else "") for w in pro_words if not w in stop_words]
        filtered_cons = [ps.stem(w).lower() + ("_con" if labels else "") for w in con_words if not w in stop_words]
        filtered_adv = [ps.stem(w).lower() + ("_adv" if labels else "") for w in adv_words if not w in stop_words]
      elif word_process == "lemma":
        # Option 3: Filter the stopwords out, lemmatize the words, and append the category 
        filtered_pros = [lemmatizer.lemmatize(w).lower() + ("_pro" if labels else "") for w in pro_words if not w in stop_words]
        filtered_cons = [lemmatizer.lemmatize(w).lower() + ("_con" if labels else "") for w in con_words if not w in stop_words]
        filtered_adv = [lemmatizer.lemmatize(w).lower() + ("_adv" if labels else "") for w in adv_words if not w in stop_words]
      elif word_process == "features":
        # Option 4: Only keep the words in the specified featureset
        filtered_pros = [ps.stem(w).lower() + ("_pro" if labels else "") for w in pro_words if w in all_features]
        filtered_cons = [ps.stem(w).lower() + ("_con" if labels else "") for w in con_words if w in all_features]
        filtered_adv = [ps.stem(w).lower() + ("_adv" if labels else "") for w in adv_words if w in all_features]
      else:
        print("Invalid word processing type. Please enter \"raw\", \"stem\", or \"lemma\". Exiting program...")
        exit()
      
      # Turn the filtered words back into a sentence and create a tuple of the review and the score
      rev = (' '.join(word for word in (filtered_pros + filtered_cons + filtered_adv)), int(score), pre_post, sampled, rev_id, company_id, industry, comp_good_bad)
      
      # Append to the processed list
      processed_revs.append(rev)

    return processed_revs

  except KeyboardInterrupt:
    print('exit')
    exit(1)

# # revs_processed is a list containing a tuple for each review. Each tuple consists of the preprocessed text for the review and the score of the review
# revs_processed = []
# for file in os.listdir(DATA_DIR + "/reviews_text/stemmed_text/"):
#   if file == ("reviews_" + before_after  + ("_swint" if swint else "_all") + ("_labels" if labels else "") + ("_gbn" if gbn else "_gb") + ("_1yr" if years_after_ipo is 1 else "") + "_thresh_" + gbn_thresh + ".pickle"):
#     revs_processed = pickle.load(open(DATA_DIR + "/reviews_text/stemmed_text/" + file, "rb"))
# if not revs_processed: 
revs_processed = preprocess_reviews(revs_raw, word_process)
#   # Save the processed reviews for later
#   pickle.dump(revs_processed, open(DATA_DIR + "/reviews_text/stemmed_text/" + "reviews_" + before_after  + ("_swint" if swint else "_all") + ("_labels" if labels else "") + ("_gbn" if gbn else "_gb") + ("_1yr" if years_after_ipo is 1 else "") + "_thresh_" + gbn_thresh + ".pickle", "wb"))


# Select the reviews to be used in terms of good/neutral/bad
if gbn_subset is "good":
  revs_processed = [rev for rev in revs_processed if rev[7] == "good"]
elif gbn_subset is "bad":
  revs_processed = [rev for rev in revs_processed if rev[7] == "bad"]
elif gbn_subset is "neutral":
  revs_processed = [rev for rev in revs_processed if rev[7] == "neutral"]
elif gbn_subset is "all":
  revs_processed = [rev for rev in revs_processed]
else:
  print("Invalid value for gbn_subset...")
  code.interact(local = locals())


# # Remove the neutral scores and change the label if we are classifying based on score
if classification_unit is "score":
#   # Get the same number of reviews pre-IPO as post-IPO so training is not baised.
#   before_revs = [rev for rev in revs_processed if rev[2] == "before"]
#   # before_revs = [(rev[0], rev[1], rev[2], 1, rev[4], rev[5], rev[6], rev[7]) for rev in before_revs]
#   after_revs = [rev for rev in revs_processed if rev[2] == "after"]
#   before_count = Counter([rev[1] for rev in before_revs])
#   after_count = Counter([rev[1] for rev in after_revs])
#   print("Number of before reviews: " + str(len(before_revs)))
#   print("Number of after reviews: " + str(len(after_revs)))

#   # Sample the after revs to get the same number as before and create a new smaller revs_processed
#   sampled_before_revs = []
#   sampled_after_revs = []
#   if before_after is "both":
#     for c in before_count:
#       if before_count[c] >= after_count[c]:
#         sampled_before_revs.append(random.sample([rev for rev in before_revs if rev[1] == c], after_count[c]))
#         sampled_after_revs.append(random.sample([rev for rev in after_revs if rev[1] == c], after_count[c]))
#       else:
#         sampled_before_revs.append(random.sample([rev for rev in before_revs if rev[1] == c], before_count[c]))
#         sampled_after_revs.append(random.sample([rev for rev in after_revs if rev[1] == c], before_count[c]))
#     sampled_before_revs = [item for sublist in sampled_before_revs for item in sublist]
#     sampled_after_revs = [item for sublist in sampled_after_revs for item in sublist]
#   else:
#     sampled_after_revs = after_revs

#   # Set the flag to 1 for the randomly sampled after revs
#   before_revs = [(rev[0], rev[1], rev[2], 1, rev[4], rev[5], rev[6], rev[7]) if rev in sampled_before_revs else rev for rev in before_revs]
#   after_revs = [(rev[0], rev[1], rev[2], 1, rev[4], rev[5], rev[6], rev[7]) if rev in sampled_after_revs else rev for rev in after_revs]
#   revs_processed = before_revs + after_revs

#   # Save the flagged results to a dataframe
#   print("Saving pandas df of processed text...")
#   revs_df = pd.DataFrame(revs_processed)
#   revs_df.columns = ['processed_rev', 'score', 'before_after', 'sampled', 'rev_id', 'company_id', 'industry', 'comp_good_bad']
#   revs_df.to_pickle(DATA_DIR + "/reviews_text/processed_text/" + "pd_processed_revs_" + before_after + ("_swint" if swint else "_all") + ("_labels" if labels else "_no_labels") + ("_bigrams" if n_gram > 1 else "_words") + ("_1yr" if years_after_ipo is 1 else "") + ".pickle")
#   revs_df.to_csv(DATA_DIR + "/reviews_text/processed_text/" + "pd_processed_revs_" + before_after + ("_swint" if swint else "_all") + ("_labels" if labels else "_no_labels") + ("_bigrams" if n_gram > 1 else "_words") + ("_1yr" if years_after_ipo is 1 else "") + ".csv")

  # Get only the flagged reviews for future processing
  revs_processed = [(rev[0], rev[1]) for rev in revs_processed]

  # Remove the reivews with a score of 3 if binary split is indicated
  if score_split is "binary":
    revs_processed = [rev for rev in revs_processed if rev[1] != 3]
    print('Reviews with score of 3 removed: ', str(len(revs_raw) - len(revs_processed)))

  # Change the scores to be "good"/"bad"/("neutral")
  if score_split != "stars":
    revs_processed = [(rev[0], "good") if rev[1] > 3 else (rev[0], "neutral") if rev[1] == 3 else (rev[0], "bad") for rev in revs_processed]

elif classification_unit is "pre_post":
  # Set the review to contain just the text and the before/after label. Split out by good/bad/neutral if specified in gbn_subset
  if gbn_subset is "good":
    revs_processed = [(rev[0], rev[2]) for rev in revs_processed if rev[7] == "good"]
  elif gbn_subset is "bad":
    revs_processed = [(rev[0], rev[2]) for rev in revs_processed if rev[7] == "bad"]
  elif gbn_subset is "neutral":
    revs_processed = [(rev[0], rev[2]) for rev in revs_processed if rev[7] == "neutral"]
  elif gbn_subset is "all":
    revs_processed = [(rev[0], rev[2]) for rev in revs_processed]
  else:
    print("Invalid value for gbn_subset...")
    code.interact(local = locals())

else:
  print("Invalid value for classification unit...")
  code.interact(local = locals())

print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
print("Tokenizing reviews...")

# Create the new document to which all of the (review, cateogry) tuples will be appended to.  
if n_gram is 1:
  documents = [(word_tokenize(rev[0]), rev[1]) for rev in revs_processed]
elif n_gram is 2:
  documents = [(list(nltk.bigrams(rev[0].split())), rev[1]) for rev in revs_processed]
else:
  print('n_gram must be specified to be 1 or 2')
  exit()

# Get just the reviews
revs_processed = [rev[0] for rev in documents]

# Create a single list containing all words from all reviews 
revs_words = [item for sublist in revs_processed for item in sublist]

# Get the frequency distribution of the words
all_words = nltk.FreqDist(revs_words)
print("Most Frequent Words:")
print(all_words.most_common(50))
print("Number of Distinct Words: " + str(len(all_words)))


# Create a new feature finding function. It returns a tuple of (features, category). Features is a dictionary containing each word in a review and a bool indicating whether it is contained in the word_features. word_features contians the 5,000 most common words from the entire text. 
print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
print("Creating featureset:")

word_features = [word[0] for word in all_words.most_common(num_features)]

def find_features(rev):
  features = {}
  for w in word_features:
    features[w] = (w in rev)
  return features

featuresets = [(find_features(rev),category) for (rev,category) in documents]


# Create the training and test sets. train_test_split creates the split, shuffles the documents, and stratifies by score to ensure there are equal proportions of scores in the training and test data. 
features = pd.DataFrame([rev[0] for rev in featuresets])
scores = [rev[1] for rev in featuresets]
x_train, x_test, y_train, y_test = train_test_split(features, scores, test_size = .1, stratify = scores)
num_train = len(y_train)
print("num_train: " + str(num_train))

# Format the data in the way the nltk wrapper requires
nltk_x_train = x_train.to_dict(orient = 'records')
nltk_x_test = x_test.to_dict(orient = 'records')
nltk_train = zip(nltk_x_train, y_train)


print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
print("featureset generated and split into training and test...")


# Create a function that predicts the scores for a binary classifier wrapped in SklearnClassifier() given the classifier has been trained.
def nltk_predict_scores(classifier, name, training_set, x_test, y_test):

  # Train the classifier
  print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
  print("Training " + name + " classifier...")
  classifier = classifier.train(training_set)

  # Save the classifier
  classifier_name = ("score_" + score_split + "_" if classification_unit is "score" else "pre_post_") + gbn_subset + "_comps_" + before_after + ("_swint_" if swint else "_all_ind_") + str(num_features) + "_feat_" + str(num_train) + "_train_" + word_process + ("_labels" if labels else "_no_labels_") + ("_words" if n_gram == 1 else "_bigrams") + "_" + name + ("_1yr" if years_after_ipo is 1 else "") + "_thresh_" + gbn_thresh + ".pickle"
  save_classifier = open(DATA_DIR + "/reviews_text/classifiers_final/" + classifier_name, "wb")
  pickle.dump(classifier, save_classifier)
  save_classifier.close()

  # Show the most informative features
  if name == "naive_bayes":
    classifier.show_most_informative_features(30)
  elif name == "logistic":
    coef = classifier._clf.coef_.tolist()[0]
    norm_coef = [float(c)/abs(max(coef)) for c in coef]
    features = training_set[0][0].keys()
    lr_top_30_features = sorted(zip(features, norm_coef), key = lambda x: abs(x[1]), reverse = True)[:30]
    print("Logistic Regression Most Important Features (normalized): " + str(lr_top_30_features))
  elif name == "random_forest":
    fi = classifier._clf.feature_importances_
    norm_fi = [float(f)/abs(max(fi)) for f in fi]
    features = training_set[0][0].keys()
    rf_top_30_features = sorted(zip(features, norm_fi), key = lambda x: -x[1])[:30]
    print("Random Forest Most Important Featues (normalized): " + str(rf_top_30_features))
  else:
    print("name not valid value input to function nltk_predict_scores: " + name)

  # Get the predicted and true values
  predicted_vals = classifier.classify_many([rev for rev in x_test])
  true_vals = y_test

  # Get the confusion matrix and accuracy
  if classification_unit is "pre_post":
    cm_labels = ["before", "after"]
  elif classification_unit is "score" and score_split is "binary":
    cm_labels = ["bad", "good"]
  elif classification_unit is "score" and score_split is "trinary":
    cm_labels = ["bad", "neutral", "good"]
  elif classification_unit is "score" and score_split is "stars":
    cm_labels = [1,2,3,4,5]
    
  else:
    print("Classification_unit and/or score_split are misspecificed. Opening debugger...")
    code.interact(local = locals())
  cm = confusion_matrix(true_vals, predicted_vals, cm_labels)
  ncm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  error = 0
  if classification_unit is "score" and score_split is "stars":
    error_values =  [[0,1,4,9,16],[1,0,1,4,9],[4,1,0,1,4],[9,4,1,0,1],[16,9,4,1,0]]
    error = 1/float(np.sum(cm))*(np.sum(np.multiply(cm, error_values)))
  acc = accuracy_score(true_vals, predicted_vals)

  # Print the results
  print(name + " accuracy percent: " + str(acc*100))
  print("Confusion Matrix: \n", cm)
  print("Normalized Confusion Matrix: ", ncm)
  if score_split is "binary" or classification_unit is not "score":
    auc = roc_auc_score(label_binarize(y_test, classes = (["bad", "good"] if classification_unit is "score" else ["before", "after"])), label_binarize(predicted_vals, classes = (["bad", "good"] if classification_unit is "score" else ["before", "after"])))
    print("AUC for ROC curve: ", str(auc))
  print(name + " analysis complete!")
  print(strftime("%Y-%m-%d %H:%M:%S", localtime()))

  return acc, cm, ncm, (auc if score_split is "binary" or classification_unit is not "score" else 0), predicted_vals, classifier

# Train and test the Naive Bayes classifier
nb_classifier = nltk.NaiveBayesClassifier
nb_acc, nb_cm, nb_ncm, nb_auc, nb_predicted_vals, nb_classifier = nltk_predict_scores(nb_classifier, "naive_bayes", nltk_train, nltk_x_test, y_test)

# # Train and test the logistic regression classifier
# lr_classifier = SklearnClassifier(LogisticRegression())
# lr_acc, lr_cm, lr_ncm, lr_auc, lr_predicted_vals, lr_classifier = nltk_predict_scores(lr_classifier, "logistic", nltk_train, nltk_x_test, y_test)

# # Train and test the random forest classifier (with 150 trees)
# rf_classifier = SklearnClassifier(RandomForestClassifier(n_estimators = 500))
# rf_acc, rf_cm, rf_ncm, rf_auc, rf_predicted_vals, rf_classifier = nltk_predict_scores(rf_classifier, "random_forest", nltk_train, nltk_x_test, y_test)

# # CODE USED TO OPEN A CLASSIFIER THAT HAS BEEN SAVED
# classifier_f = open(DATA_DIR + "/reviews_text/classifier_score/score_trinary_bothall_ind_6000_feat_14431_train_stem_labels_bigramsnaive_bayes.pickle","rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()

def get_most_informative_features(nb_clf, n = 100):
  # Determine the most relevant features, and display them.
  mif = []
  cpdist = nb_clf._feature_probdist
  print('Most Informative Features')

  for (fname, fval) in nb_clf.most_informative_features(n):
    def labelprob(l):
      return cpdist[l, fname].prob(fval)

    labels = sorted([l for l in nb_clf._labels
                     if fval in cpdist[l, fname].samples()],
                    key=labelprob)
    if len(labels) == 1:
      continue
    l0 = labels[0]
    l1 = labels[-1]
    if cpdist[l0, fname].prob(fval) == 0:
      ratio = 'INF'
    else:
      ratio = (cpdist[l1, fname].prob(fval) / cpdist[l0, fname].prob(fval))
    # print(('%24s = %-14r %6s : %-6s = %s : 1.0' %
    #        (fname, fval, ("%s" % l1)[:6], ("%s" % l0)[:6], ratio)))
    mif.append((fname, fval, l1, l0, ratio))

  return mif 

# Get and save the most informative features
nb_mif = get_most_informative_features(nb_classifier)
nb_mif = pd.DataFrame(nb_mif)


nb_mif.columns = ['words', 'fval', 'lab1','lab2', 'ratio']
mif_file_name = "mif_" + ("score_" + score_split + "_" if classification_unit is "score" else "pre_post_") + gbn_subset + "_comps_" + before_after + ("_swint_" if swint else "_all_ind_") + str(num_features) + "_feat_" + str(num_train) + "_train_" + word_process + ("_labels" if labels else "_no_labels_") + ("_words" if n_gram == 1 else "_bigrams") + "_"  + "_naive_bayes" + ("_1yr" if years_after_ipo is 1 else "") + "_thresh_" + gbn_thresh + ".pickle"
nb_mif.to_pickle(DATA_DIR + "/reviews_text/temp2/" + mif_file_name)






print("Code complete.")
print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
code.interact(local = locals())


