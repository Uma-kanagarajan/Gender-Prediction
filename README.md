# Gender-Prediction using ML model

#### Build a machine learning model to predict the gender based on user given names.

> **Dataset:** Given CSV file contains names and its respective gender.

> **Project Objective:** Develop a model to predict the gender of a user given names.

> **Project Approach:** Given name is tokenized into characters and fed into a neural network that gives the output of gender probability. Gender is predicted based on the probability threshold value of 0.5.

> **Files Provided:**
  1. Requirements.txt file
  2. name_gender_prediction.ipynb file and the same in python file as name_gender_prediction.py (These files generate only base model)
  3. name_gender_prediction_keras_tuner.ipynb file and the same in python file as name_gender_prediction_keras_tuner.py (These files generate base model and hyperparameter tuned model)
  4. gender_predict_app.py (For web app)
  5. gender_prediction.h5 (my trained model)

> **Gender prediction Web app link**
Please click the provided [gender prediction webapp link](https://name-gender-prediction.herokuapp.com/). This web app is developed using python streamlit package and deployed in heroku.

> **Procedures to be followed to run this project**
1. Ensure all the files are downloaded and saved in the same directory
2. Open *name_gender_prediction.ipynb* to run it in colab.This file will generate a trained model.
3. Open *name_gender_prediction.py* in your IDE if you prefer to run it locally.
4. Ensure all the necessary packages are installed as stated in requirements.txt
5. Open *gender_predict_app.py* and run it with a terminal command 'streamlit run gender_predict_app.py' to get a localhost web page.

> **Note:**
Hyperparameter tuner code is provided but due to limits of GPU, code got stopped after 300 iterations and hence i didn't get hyperparameter tuned model and its respective parameters. Due to this issue, I have provided separate .ipynb and .py files for base model and tuner model.

> **Conclusion:**  
> *Base model metrics*  
>  Training accuracy: 90%   
>  Testing accuracy: 89%  
>  This is the highest evaluation metrics that I have achieved for base model without overfitting the data.
