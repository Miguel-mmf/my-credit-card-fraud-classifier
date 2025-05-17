# Model Card

Model cards are a succinct approach for documenting the creation, use, and shortcomings of a model. The idea is to write a documentation such that a non-expert can understand the model card's contents. For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

Credit card fraud occurs when someone steals or uses a credit card's information without the cardholder's permission. To combat and prevent fraudulent transactions, credit card companies and financial institutions have implemented various measures. Most modern solutions leverage artificial intelligence (AI) and machine learning (ML).

The purpose of this project is to emulate a service used by one of these institutions to predict whether a purchase is fraudulent. The service receives as input all the information about a purchase made with a credit card by a client and returns as output the probability that the purchase is fraudulent, as well as a recommendation on whether it should be flagged as fraud. The response from this service can be used to prevent customers from being charged for items they did not purchase

## Model Details

@Miguel-mmf created the model. A complete data pipeline was built using DVC and Scikit-Learn to train a XGBoost model. For the sake of understanding, a simple hyperparameter-tuning was conducted, and the hyperparameters values adopted in the train are described in a yaml file.

## Intended Use


## Training Data

The dataset used in this project is based on individual income in the United States. The data is from the 1994 census, and contains information on an individual's marital status, age, type of work, and more. The target column, or what we want to predict, is whether individuals make less than or equal to 50k a year, or more than 50k a year.

You can download the data from the University of California, Irvine's website.

After the EDA stage of the data pipeline, it was noted that the training data is imbalanced when considered the target variable and some features (sex, race and workclass.


## Evaluating Data

The dataset under study is split into Train and Test during the Segregate stage of the data pipeline. 70% of the clean data is used to Train and the remaining 30% to Test. Additionally, 30% of the Train data is used for validation purposes (hyperparameter-tuning). This configuration is done in a yaml file.

## Metrics

In order to follow the performance of machine learning experiments, the project marked certains stage outputs of the data pipeline as metrics. The metrics adopted are: accuracy, f1, precision, recall.

The follow results will be shown:

table

## Ethical Considerations

We may be tempted to claim that this dataset contains the only attributes capable of predicting someone's income. However, we know that is not true, and we will need to deal with the class imbalances somehow.


## Caveats and Recommendations

It should be noted that the model trained in this project was used only for validation of a complete data pipeline. It is notary that some important issues related to dataset imbalances exist, and adequate techniques need to be adopted in order to balance it.
