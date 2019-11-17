from app import app
from app import mlCategoriser
import json
import _json
from flask import render_template
import tensorflow as tf
import os
from keras.preprocessing import image

import numpy as np

@app.route("/")
def index():
    return render_template("public/index.html")

@app.route("/Results")
def doit():
    # load json and create model
    with open('model02.json') as json_file:
        data = json.load(json_file)
    loaded_model = tf.keras.models.model_from_json(data)
    # load weights into new model
    loaded_model.load_weights("model02.h5")
    print("Loaded model from disk")

    class_indices = ["10_Choco_Haps", "11_K_Classic_Mehl", "12_K_Classic_Zucker",
                     "1_Haribo_Goldbaer", "2_Chock_IT", "4_Kornflakes",
                     "5_K_Classic_Paprika_Chips", "6_Birnen_Dose", "7_Pfirsiche_Dose",
                     "9_Aprikosen_Dose"]

    test_image = image.load_img('app/static/img/Compare.jpg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = loaded_model.predict(test_image)
    print(result)
    return class_indices[np.argmax(result)]

@app.route("/about")
def about():
    return """
    <h1 style='color: red;'>I'm a red H1 heading!</h1>
    <p>This is a lovely little paragraph</p>
    <code>Flask is <em>awesome</em></code>
    """
@app.route("/jinja")
def jinja():

    # Strings
    my_name = "Julian"

    # Integers
    my_age = 30

    # Lists
    langs = ["Python", "JavaScript", "Bash", "Ruby", "C", "Rust"]

    # Dictionaries
    friends = {
        "Tony": 43,
        "Cody": 28,
        "Amy": 26,
        "Clarissa": 23,
        "Wendell": 39
    }

    # Tuples
    colors = ("Red", "Blue")

    # Booleans
    cool = True

    # Classes
    class GitRemote:
        def __init__(self, name, description, domain):
            self.name = name
            self.description = description
            self.domain = domain

        def pull(self):
            return f"Pulling repo '{self.name}'"

        def clone(self, repo):
            return f"Cloning into {repo}"

    my_remote = GitRemote(
        name="Learning Flask",
        description="Learn the Flask web framework for Python",
        domain="https://github.com/Julian-Nash/learning-flask.git"
    )

    # Functions
    def repeat(x, qty=1):
        return x * qty

    return render_template(
        "public/jinja.html", my_name=my_name, my_age=my_age, langs=langs,
        friends=friends, colors=colors, cool=cool, GitRemote=GitRemote,
        my_remote=my_remote, repeat=repeat
    )