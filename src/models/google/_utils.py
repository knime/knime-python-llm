import knime.extension as knext

from ..base import model_category

# Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
KNOWN_DEPRECATED_MODELS = [
    "gemini-ultra",
    "gemini-ultra-vision",
    "gemini-pro",
    "gemini-pro-vision",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-001",
    "gemini-1.0-pro-001",
    "gemini-1.0-pro-002",
    "gemini-1.0-pro-vision-latest",
    "gemini-1.0-flash-vision-001",
    "text-bison",
    "chat-bison",
    "code-gecko",
    "textembedding-gecko-multilingual@001",
    "textembedding-gecko@003",
    "textembedding-gecko@002",
    "textembedding-gecko@001",
]

# References:
# https://ai.google.dev/gemini-api/docs/models
# https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings#supported-models
GEMINI_CHAT_MODELS_FALLBACK = [
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]
VERTEX_AI_GEMINI_EMBEDDING_MODELS_FALLBACK = [
    "text-embedding-large-exp-03-07",
    "text-embedding-005",
]
GOOGLE_AI_STUDIO_GEMINI_EMBEDDING_MODELS_FALLBACK = [
    "gemini-embedding-exp-03-07",
    "text-embedding-005",
]

# icons
vertex_ai_icon = "icons/google/vertex_ai.png"
google_ai_studio_icon = "icons/google/google_ai_studio.png"
google_icon = "icons/google/google_super_g.png"
gemini_icon = "icons/google/gemini.png"

google_category = knext.category(
    path=model_category,
    name="Google",
    level_id="google",
    description="Google GenAI",
    icon=google_icon,
)

DEFAULT_VERTEX_AI_LOCATION = "us-central1"


# Taken from https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
def _vertex_ai_location_choices_provider(ctx):
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
