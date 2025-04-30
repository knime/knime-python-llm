import knime.extension as knext
from ..base import model_category

anthropic_icon = "icons/generic/brain.png"
anthropic_category = knext.category(
    path=model_category,
    name="Anthropic",
    level_id="anthropic",
    description="Anthropic models",
    icon=anthropic_icon,
)
