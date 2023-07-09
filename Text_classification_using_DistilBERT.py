# Install or upgrade ktrain library
pip install --upgrade ktrain

# Commented out IPython magic to ensure Python compatibility.
%reload_ext autoreload
%autoreload 2
%matplotlib inline

# Import necessary libraries
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import ktrain
from ktrain import text
from sklearn.datasets import fetch_20newsgroups

# Define categories for the dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med', 'rec.sport.baseball']

# Fetch the training and testing datasets
train = fetch_20newsgroups(
    subset='train',
    categories=categories,
    shuffle=True,
    random_state=0
)

test = fetch_20newsgroups(
    subset='test',
    categories=categories,
    shuffle=True,
    random_state=0
)

print(test.keys())

# Split the data into features and labels
X_train = train.data
y_train = train.target

X_test = test.data
y_test = test.target

# Build ML model with Transformer
model_name = 'distilbert-base-uncased'
trans = text.Transformer(model_name, maxlen=512, class_names=categories)

train_data = trans.preprocess_train(X_train, y_train)
test_data = trans.preprocess_test(X_test, y_test)

model = trans.get_classifier()

learner = ktrain.get_learner(model, train_data=train_data, val_data=test_data, batch_size=16)

learner.fit_onecycle(1e-4, 1)

learner.validate()

learner.view_top_losses(n=5, preproc=trans)

# Predict on new data

predictor = ktrain.get_predictor(learner.model, preproc=trans)

x = 'I have a 30-year-old friend who is going through some difficulties in his life due to a disease.'

predictor.predict(x)

predictor.save('model')
