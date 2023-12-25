# Vertex AI SDK - uncomment below & run
# pip3 install --upgrade --user google-cloud-aiplatform
# gcloud auth application-default login

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image

import http.client
import typing
import urllib.request

# create helper function
def load_image_from_url(image_url: str) -> Image:
    with urllib.request.urlopen(image_url) as response:
        response = typing.cast(http.client.HTTPResponse, response)
        image_bytes = response.read()
    return Image.from_bytes(image_bytes)

image_url = load_image_from_url(
    "https://storage.googleapis.com/cloud-samples-data/vertex-ai/llm/prompts/landmark1.png"
)

def generate_text(project_id: str, location: str) -> str:
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    # Load the model
    multimodal_model = GenerativeModel("gemini-pro-vision")
    # Query the model

    response = multimodal_model.generate_content(
        [
            image_url,
            # Add an example query
            "what is shown in this image?",
        ]
    )
    print(response)
    return response.text

project_id = "INSERT YOUR ID PROJECT"
location = "us-central1"
generate_text(project_id, location)