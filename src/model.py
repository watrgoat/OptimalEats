# model.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot

def RecipeEmbeddingModel(num_recipes, embedding_size=50):
    """
    A simple embedding model for learning recipe embeddings.
    """
    recipe_input = Input(name='recipe_input', shape=[1])
    recipe_embedding = Embedding(name='recipe_embedding', 
                                 input_dim=num_recipes, 
                                 output_dim=embedding_size)(recipe_input)
    recipe_vec = Flatten(name='flatten_recipe')(recipe_embedding)

    model = Model(inputs=[recipe_input], outputs=recipe_vec)

    model.compile(optimizer='adam', loss='rmse')
    
    return model