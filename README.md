# Gender-Prediction using ML model

#### Build a machine learning model to predict the gender based on user given names.

Install all the necessary packages mentioned in the requirement.txt file

Check the given dataset

Load the dataset

Preprocess the dataset using preprocess_name function which takes names as input and converts it into an array using below steps.
1. Removes non-alphabetical characters and spaces from the given input name.
2. Then tokenize each characters of the name into a vector and converts it into an integer
3. Given name length gets changed to a fixed length of 20 with the help of post padded sequence

Label the gender column of input dataset into binary values. '0' for female and '1' for male

Split the dataset into training set and test set

Base model creation
LSTM model is used with
1. Input dimension as 28(Total number of alphabets + 1)
2. Output dimension as 28.
3. Name length is fixed to 20
4. callback is used for early stop
5. Fit and save the model
6. Visualize the model accuracy for test and validation dataset

Model is defined using sequential API with the layers such as Embedding, LSTM, Dropout, Dense and a final dense output layer.
Adam optimizer, Binary crosentropy are considered.

Evaluate the model using test set

Predict the gender by getting input from the user

Hyperparameter tuner using keras tuner


