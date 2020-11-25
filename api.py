import traceback
from flask import render_template, request, redirect, url_for
from flasgger import Swagger
import logging.config

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from src.predict import generate_prediction

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize the Flask application
app = Flask(__name__, template_folder="app/templates")

# Configure flask app from flask_config.py
app.config.from_pyfile("src/config.py")


@app.route("/")
def index():
    """Main view that lists the app's web page.
    Create view into index page that servers as the platform to input data in order to
    generate a prediction.
    Returns: rendered html template
    """

    return render_template("index.html")


@app.route("/add", methods=["GET", "POST"])
def add_entry():
    """View that process a POST with new text input
    Returns: predicted label
    """

    text_input = request.form["input_text"]

    # Generate prediction result
    logger.info("Generating prediction.")

    try:
        label = generate_prediction(text_input)

        return render_template("index.html", input=text_input, result=label)

    except:
        traceback.print_exc()
        logger.warning("Not able to display prediction, error page returned")
        return render_template("error.html", result="Result not available")


if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"], host=app.config["HOST"])
