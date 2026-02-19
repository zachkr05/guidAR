# robot_path_intent_classifier.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from classifierData import *

MODIFYTRAJ = "MODIFY TRAJECTORY"
GENTARGETPOINT = "GENERATE TARGET POINT"

class IntentClassifier:
    def __init__(self):
        self.all_data = training_data + edge_cases + contextual_examples
        texts, labels = zip(*self.all_data)

        self.label_to_num = {
            MODIFYTRAJ: 0,
            GENTARGETPOINT: 1
        }
        self.num_to_label = {0: MODIFYTRAJ, 1: GENTARGETPOINT}

        y = [self.label_to_num[label] for label in labels]

        X_train, _, y_train, _ = train_test_split(
            texts, y, test_size=0.2, random_state=42, stratify=y
        )

        self.classifier = make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=5000,
                stop_words='english',
                lowercase=True
            ),
            LogisticRegression(random_state=42, max_iter=1000)
        )

        self.classifier.fit(X_train, y_train)

    def classify_intent(self, text):
        prediction = self.classifier.predict([text])[0]
        confidence = max(self.classifier.predict_proba([text])[0])
        result = self.num_to_label[prediction]
        return result, confidence