# CS412-PROJECT

Abbas Ali Hakeem  
Alaa Almouradi  
M Safwan Yasin  
Omar Mirza  
Saleh Alshurafa

## Overview of the Repository

### Folder structure:

The repository has the data used in the python notebook along with the base notebook and our notebook.

```
|
|-data/
|-plots/
|-CS412Project.ipynb
|-hw_score_predict.ipynb
```

### Outside scripts

The base notebook's text extraction from the html file was used.

## Methedology

### Feature extraction

After using the tools given in the base notebook to get the list of prompts and responses we extracted the following features:

#### 1. Question Presence:

- vectorize the questions of the assignment after cleaning them

This is the essence of the code responsible:

```python
questions = [ i.lower().translate(str.maketrans('','',string.punctuation+"\n")) for i in questions]

questionsVectorized = [spacynlp(question).vector for question in questions]
```

- for each conversation:

  - get number of prompts
  - get average length of prompt
  - for each response from chatgpt:
    - clean it
    - vectorize it
    - measure similarity between response and questions
    - add the similarity to each question as a column in the row

  The essence:

  ```python
  for question in range(9):
      studentRow["Q"+str(question+1)+" presence"][0] = (1.0 /(1+ distance.euclidean(responseVector, questionsVectorized[question]))) / len(responses)
  ```

  The similarity metric here is using an inverse of euclidean distance that makes the similarity between two vectors in the range ( 0, 1 ].

#### 2. Representative similarity (unused but implemented):

- sort the scores
- make a representative vector for each 20% segment in the ordered scores
- for each conversation:
  - clean it
  - vectorize it
  - measure similarity between response and the 5 representative vectors
  - add the similarity to each representative vector as a column in the row
  The essence:
  ```python
  for k in range(5):
      studentRow["top "+str(k+1)+" fifth similarity"][0] = (1.0 /(1+ distance.euclidean(studentVector, pentileVectors[k])))
  ```
  Again, using euclidean similarity.

#### 3. Keyword count:

- for each conversation:
  - count selected keywords in prompts and responses and use the count for each keyword as a feature

keywords (p for prompts and r for response):

```python
keywords2search_p = [r'\bhypothetical\b', r'warning', r'error:', r'wrong', r'thank', r'\bentropy\b', r'\bno\b', r'\bpreprocess\b', r'hyperparameters', r're-train', r'\w?health_metrics\w?', r'how']

keywords2search_r = [r'thank', r'\bhypothetical\b', r'\bentropy\b', r'apologize', r'\bmissing\b', r'hyperparameters', r'\w?health_metrics\w?']
```

#### 4. Calculated features:

- features calculated from previous features

The essence:

```python
# Combined chat stats
shuffledFrame['combined_chat_stats'] = shuffledFrame['prompt_avg_chars'] * shuffledFrame['#user_prompts'] + shuffledFrame['response count'] * shuffledFrame['avg. response length']
shuffledFrame['combined_chat_stats'].fillna(0, inplace=True)
# Combined questions
shuffledFrame['combined_questions'] = shuffledFrame['Q1 presence'] + shuffledFrame['Q2 presence'] + shuffledFrame['Q3 presence'] + shuffledFrame['Q4 presence'] + shuffledFrame['Q5 presence'] + shuffledFrame['Q6 presence'] + shuffledFrame['Q7 presence'] + shuffledFrame['Q8 presence'] + shuffledFrame['Q9 presence']
# Combined keywords feature
shuffledFrame['combined_kw'] = shuffledFrame['#user_prompts'] + shuffledFrame['#thank'] + shuffledFrame['#\w?health_metrics\w?'] + shuffledFrame['#re-train'] + shuffledFrame['#\\bpreprocess\\b'] + shuffledFrame['#\\bmissing\\b'] + shuffledFrame['#wrong'] + shuffledFrame['#\\bno\\b']

```

### Train-test split

since the data is continous, shuffling the data before splitting would deal with the skewed nature of the data.

```python
shuffledFrame = shuffle(FinalDataFrame, random_state=42)

Xframe = shuffledFrame.drop(columns=['code','grade'])
Yframe = shuffledFrame['grade']

train_data, test_data, train_labels, test_labels = train_test_split(Xframe, Yframe, test_size=0.2, random_state=42)
```

### Models

Our approach with the models was to try out multiple models with
different combinations of hyper parameters to see which perform best.
In order to decide what performs best, we looked at the Mean Squared
Error value produced by the model for the test data.

#### **The models we tried were**:

- Decision Tree
- K-NN
- Gradient Boosting
- Random Forest
- Neural Networks

For Decision Tree, Gradient Boosting and Random Forest, the hyper
parameters we focused on were the max_depth and random_state
parameters.

- max_depth: the maximum depth of the tree.

- random_state: controls the randomness of the estimator.

For the K-NN, the hyper parameter we focused on was the n_neighbors.

- n_neighbors: the value n for the number of neighbors.

For the Neural Network we considered the hidden_layer_size and random_state
hyper parameters.

- hidden_layer_size: the number of neurons in the ith layer.

#### **How hyper parameter tuning was performed**

At first, we used the GridSearchCV function from the scikit-learn library.
However, it failed to give us the most optimal results in terms of minimizing
MSE. Additionally, it took over 40 minutes to fully run the code for testing
Gradient Boosting. Therefore, another approach was taken to tune the hyper parameters.

The other approach involved using simple for loops. In the models were we
focused on more than one hyper parameter, we used nested loops to loop
over a range of values for each hyper parameter. A combination of hyper
parameter values would then be used to train a model and obtain its MSE
over test data. The combination that produced the least MSE on test data
were selected.

In the code block below, an example can be seen of how the hyper parameter tuning was
done for Gradient Boosting. A similar approach was used in the other models.

```python
min_mse_train = 1000
min_mse_test = 1000
min_r2_train = 1000
min_r2_test = 1000

for rs in range(50):
    for md in range(1, 51):

        regressor = GradientBoostingRegressor(random_state=rs,criterion='squared_error', max_depth=md)
        regressor.fit(train_data, train_labels)

        y_train_pred = regressor.predict(train_data)
        y_test_pred = regressor.predict(test_data)

        mse_train = mean_squared_error(train_labels, y_train_pred)
        mse_test = mean_squared_error(test_labels, y_test_pred)
        r2_train = r2_score(train_labels, y_train_pred)
        r2_test = r2_score(test_labels, y_test_pred)

        if(mse_test < min_mse_test):
            best_rs = rs
            best_md = md
            min_mse_train = mse_train
            min_mse_test = mse_test
            min_r2_train = r2_train
            min_r2_test = r2_test
```

#### **After hyper parameter tuning**

The hyper parameter combination that we found from the previous step are
then used to train a model and to check the final results. We did that
because the MSE and R2 results produced during the tuning process were
not very accurate and we would see much improved results when using these
hyper parameters on a model in a separate code block.

The code block below shows an example of how models were trained and tested.
This example shows the Gradient Boosting trained with the best parameters we
obtained from the hyper parameter tuning.

```python
regressor = GradientBoostingRegressor(random_state=45,criterion='squared_error', max_depth=9)
regressor.fit(train_data, train_labels)

# Prediction
y_train_pred = regressor.predict(train_data)
y_test_pred = regressor.predict(test_data)

# Calculation of Mean Squared Error (MSE)accuracy_score, maccuracy_score, mean__
print("MSE Train:", mean_squared_error(train_labels,y_train_pred))
print("MSE TEST:", mean_squared_error(test_labels,y_test_pred))

print("R2 Train:", r2_score(train_labels,y_train_pred))
print("R2 TEST:", r2_score(test_labels,y_test_pred))
```

## Results

After all the operations we carried out on the data and after tuning and testing
different models, we arrived at the following results.

Using the data modified by the various operations we carried out, the
Gradient Boostig model had the best performances. The model produced
the lowest MSE and the highest R2 on test data. Our second best model,
Decision Tree model, produced results that were close but not as good
as Gradient Boosting. The worst performing model was Neural Network.

The table below shows the performance of all the models we made. Note that the
base model mentioned here was **not run on modified data**. The results displayed
for it are the ones it had when we first started working on the base code.
It is displayed just so that we show the overall improvement we have
we have achieved in this project.

| Model              | Hyper Parameters                            | MSE on training data | MSE on test data | R2 on training data | R2 on test data |
| ------------------ | ------------------------------------------- | -------------------- | ---------------- | ------------------- | --------------- |
| Base Decision Tree | random_state=0 and max_depth=10             | 8.28                 | 101.55           | 0.95                | 0.095           |
| Gradient Boosting  | random_state=45 and max_depth=9             | 0.0000003            | 18.5             | 0.99                | 0.25            |
| Decision Tree      | random_state=45 and max_depth=10            | 0.18                 | 19.2             | 0.99                | 0.2             |
| K-NN               | n_neighbors=68                              | 247.38               | 43.44            | 0.06                | -0.08           |
| Random Forest      | random_state=23 and max_depth=1             | 164.7                | 40.9             | 0.38                | -0.77           |
| Neural Network     | random_state=42 and hidden_layer_size=(31,) | 467.1                | 181.37           | -0.77               | -6.84           |

#### **Actual vs Predictied Grade Plots for Our Models**

1. Gradient Boosting

   ![Alt text](plots\gradient_boosting.png)

2. Decision Tree

   ![Alt text](plots\decision_tree.png)

3. K-NN

   ![Alt text](plots\knn.png)

4. Random Forest

   ![Alt text](plots\random_forest.png)

5. Neural Network

   ![Alt text](plots\neural_network.png)

## Contributions

**Alaa & Safwan**: laid the first pipeline which was modified gradually across the project, implemented the questions presence feature extraction, and made plots of predicition vs actual for each model

**Omar & Abbas**: implemented the keywords feature extraction and the calculated features

**Saleh**: implemented the representative vectors and did model selection & hyper parameter tuning
