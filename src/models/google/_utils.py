import knime.extension as knext

from ..base import model_category

from typing import List

DEFAULT_VERTEX_AI_LOCATION = "europe-west1"

# Taken from https://ai.google.dev/gemini-api/docs/models
GEMINI_CHAT_MODELS_FALLBACK = [
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.0-flash",
    "gemini-2.5-pro-exp-03-25",
]
GEMINI_EMBEDDING_MODELS_FALLBACK = ["gemini-embedding-exp", "models/text-embedding-004"]

vertex_ai_icon = "icons/vertex_ai.png"
google_ai_studio_icon = "icons/google_ai_studio.png"
google_icon = "icons/google_super_g.png"

google_category = knext.category(
    path=model_category,
    name="Google",
    level_id="google",
    description="Google GenAI",
    icon=google_icon,
)


# Taken from https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
def _vertex_ai_location_choices_provider(ctx) -> List[str]:
    return [
        # United States
        "us-central1",
        "us-east1",
        "us-east4",
        "us-east5",
        "us-south1",
        "us-west1",
        "us-west4",
        # Canada
        "northamerica-northeast1",
        # South America
        "southamerica-east1",
        # Europe
        "europe-central2",
        "europe-north1",
        "europe-southwest1",
        "europe-west1",
        "europe-west2",
        "europe-west3",
        "europe-west4",
        "europe-west6",
        "europe-west8",
        "europe-west9",
        # Asia Pacific
        "asia-east1",
        "asia-east2",
        "asia-northeast1",
        "asia-northeast3",
        "asia-south1",
        "asia-southeast1",
        "australia-southeast1",
        # Middle East
        "me-central1",
        "me-central2",
        "me-west1",
    ]
