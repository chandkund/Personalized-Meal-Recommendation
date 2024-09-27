# 🍽️ Personalized Meal Recommendation System

This project focuses on building a personalized meal recommendation system by predicting cuisines based on the ingredients used. Leveraging NLP and a deep learning model, this system helps users discover meal options tailored to their preferences.

## Project Overview

In this project, we classify cuisines based on ingredients using an LSTM-based neural network. The model is trained to predict the cuisine type given a list of ingredients, making it a useful tool for personalized meal recommendations. The system can be extended to provide suggestions for meal planning and grocery shopping.

## Dataset Description

- **Ingredients:** The ingredients used in the recipe. This is a text-based feature that requires preprocessing.
- **Cuisine:** The cuisine type (e.g., Italian, Indian, Mexican). This is the target variable for classification.

Sample data:

| Description        | Quantity | InvoiceDate        | UnitPrice | CustomerID | Country |
|--------------------|----------|--------------------|-----------|------------|---------|
| WHITE METAL LANTERN| 6        | 12/1/2010 8:26     | 3.39      | 17850.0    |  India  |
| CREAM CUPID HANGER | 8        | 12/1/2010 8:26     | 2.75      | 17850.0    |  India  |

## Model Architecture

We use a deep learning model with the following architecture:

- **Embedding Layer:** Converts words into dense vector representations.
- **Bidirectional LSTM:** To capture dependencies in both forward and backward directions.
- **Dense Layer:** Fully connected layer for further processing.
- **Softmax Layer:** Output layer with softmax activation to classify cuisines.

```python
model = Sequential([
    layers.Embedding(voc_size, 40),
    layers.Bidirectional(layers.LSTM(100)),
    layers.Dense(100, activation='relu'),
    layers.Dense(len(le.classes_), activation='softmax')
])
```

## Preprocessing

- **Tokenization:** Breaking down the ingredients into individual tokens.
- **Lemmatization:** Reducing words to their base form.
- **Padding:** Ensuring that each sequence has a uniform length for input to the LSTM.

```python
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

def transform_data(text):
  # Lemmatization and tokenization process
  pass

df['ingredients_transformed'] = df['ingredients'].apply(transform_data)
```

## Training and Evaluation

- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Evaluation Metric:** Accuracy

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
```


## Future Enhancements

- Introduce a more advanced recommendation system based on user preferences.
- Integrate with grocery delivery APIs for a seamless shopping experience.
- Improve model accuracy by experimenting with different architectures.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/chandkund/Personalized-Meal-Recommendation.git
   cd Personalized-Meal-Recommendation
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:
   ```bash
   python Personalized meal.ipynb
   ```

## License

This project is licensed under the MIT License.

👨‍🍳 Happy Cooking
