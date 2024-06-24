
# Lung Cancer Prediction Using Machine Learning Models
This project predicts lung cancer using machine learning models: SVM, Logistic Regression, KNN, AdaBoost, and Random Forest. It includes data preprocessing, model training, evaluation, and comparative analysis of model accuracy. The project offers Jupyter notebooks, trained models, and visualizations.





## Introduction
Lung cancer is one of the most common and deadly forms of cancer. Early detection is crucial for effective treatment. This project aims to leverage machine learning algorithms to create predictive models that can assist in the early detection of lung cancer.
## Project Structure
data/: Contains the dataset used for training and testing the models.
notebooks/: Jupyter notebooks containing the code for data preparation, model implementation, and evaluation.
models/: Serialized versions of the trained models.
results/: Visualizations and evaluation metrics for each model.
README.md: Project description and setup instructions.
requirements.txt: List of dependencies required to run the project.

## Installation
1. Clone the repository:
git clone https://github.com/yourusername/lung-cancer-prediction.git

2. Navigate to the project directory:
cd lung-cancer-prediction

3. Install the required dependencies
pip install -r requirements.txt

## Usage
Run the Jupyter notebooks in the notebooks/ directory to see the data preparation, model training, and evaluation processes.

to train the models and make predictions, use the following commands within the notebooks:
# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Logistic Regression Accuracy:', accuracy)

# SVM
from sklearn.svm import SVC
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('SVM Accuracy:', accuracy)

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('KNN Accuracy:', accuracy)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=100, random_state=42)
abc.fit(X_train, y_train)
y_pred = abc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('AdaBoost Accuracy:', accuracy)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print('Random Forest Accuracy:', accuracy)


Use the provided code to save and load the trained model:

import pickle

filename = "Trained_model.sav"
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open("Trained_model.sav", 'rb'), encoding='utf-8')

Use the prediction function to predict lung cancer status:

def predict(model, inputs):
    input_array = []
    for key in inputs.keys():
        if key != 'LUNG_CANCER':
            input_array.append(inputs[key])
    prediction = model.predict([input_array])
    return prediction[0]

prediction = predict(model, inputs)
print("The model predicts that the person's lung cancer status is:", prediction)


## results
The project provides accuracy scores for each model, along with visualizations of the data and performance metrics. Comparative analysis helps in understanding the strengths and weaknesses of each model.
## Contributing
Feel free to explore, use, and contribute to this project. If you have any questions or suggestions, please open an issue or submit a pull request.
## License
This project is licensed under the MIT License - see the LICENSE file for details.
