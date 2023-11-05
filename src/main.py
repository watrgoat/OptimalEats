# main.py
import os
import pandas as pd
from model import RecipeEmbeddingModel
from sklearn.model_selection import train_test_split

# Load the dataset

df = pd.read_pickle("../data/processed/recipes.pkl")

# Data processing functions
def filter_recipes_by_diet(df, dietary_preferences):
    """
    Filters recipes based on dietary preferences provided.
    :param df: DataFrame with recipes
    :param dietary_preferences: dict with dietary flags and boolean values
    :return: Filtered DataFrame
    """
    # can still have if false

    filtered_df = df
    
    if dietary_preferences['vegetarian']:
        filtered_df = filtered_df[filtered_df['vegetarian'] == True]

    if dietary_preferences['vegan']:
        filtered_df = filtered_df[filtered_df['vegan'] == True]

    if dietary_preferences['glutenFree']:
        filtered_df = filtered_df[filtered_df['glutenFree'] == True]

    if dietary_preferences['dairyFree']:
        filtered_df = filtered_df[filtered_df['dairyFree'] == True]

    if dietary_preferences['cookingMinutes'] < 150:
        filtered_df = filtered_df[filtered_df['cookingMinutes'] <= dietary_preferences['cookingMinutes']]
    
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df

# TensorFlow does not handle string inputs well, so we need to preprocess the dataset
# to convert strings into integer indices
def preprocess_data(df):
    # convert cuisines and titles into indices
    df['recipe_index'] = df['recipe_id'].astype('category').cat.codes
    df['cuisine_index'] = df['cuisine'].astype('category').cat.codes
    return df

# Assume we have a function that can convert cuisines and titles into indices
df = preprocess_data(df)

# Splitting data into training and testing sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Prepare the input data for the model, assume the function `convert_to_model_input`
# can take the dataframe and build the necessary input arrays
train_input = convert_to_model_input(train)
test_input = convert_to_model_input(test)

# Number of unique recipes
num_recipes = df['recipe_index'].nunique()

# Instantiate the model
model = RecipeEmbeddingModel(num_recipes)

# Train the model
model.fit(train_input, train['rating'], epochs=10, validation_split=0.1)

# Save the model for later use
model.save('recipe_embedding_model.h5')

# Add more functions to handle user input and recommendations

if __name__ == '__main__':
    # Handle user inputs and call the relevant functions
    pass
