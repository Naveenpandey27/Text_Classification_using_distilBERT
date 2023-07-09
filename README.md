# Text_Classification_using_distilBERT
This project showcases text classification using the ktrain library. It trains a machine learning model based on the 'distilbert-base-uncased' transformer and applies it to classify new text data into predefined categories. The project uses the 'fetch_20newsgroups' dataset from scikit-learn, which contains news articles from various domains.

**Installation**
To run the code, follow these steps:

Install or upgrade the ktrain library by running the following command:

**pip install --upgrade ktrain**

Make sure to have the necessary dependencies installed. The code relies on the scikit-learn library, which should be installed by default in most Python environments.

Copy and paste the provided code into your Python environment, such as a Jupyter Notebook or a Python script.

Execute the code and observe the results.

Usage
1 - The code starts by importing the required libraries, including ktrain, TensorFlow, and scikit-learn.

2 - It defines the categories or labels for the dataset, which are 'alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med', and 'rec.sport.baseball'. These categories represent different topics or domains for the text classification task.

3 - The code fetches the training and testing datasets using the 'fetch_20newsgroups' function from scikit-learn. The datasets are split into subsets and shuffled for randomness.

4 - Next, the data is preprocessed. The text data is split into features (X) and labels (y) for both the training and testing datasets.

5 - A machine learning model is built using the 'distilbert-base-uncased' transformer. This model is capable of understanding and classifying text.

6 - The model is trained using the training dataset and validated using the testing dataset. The 'fit_onecycle' method is used to train the model with a one-cycle learning rate schedule.

7 - The model's performance is evaluated by viewing the top losses, which displays the samples with the highest prediction errors.

8 - Finally, the model is used to make predictions on new text data. A specific example sentence is provided, and the model predicts the most suitable category for it.

Contact
For any inquiries or questions, please contact **naveenpandey2706@gmail.com.**

Please don't hesitate to reach out if you have any feedback or suggestions for this project.
