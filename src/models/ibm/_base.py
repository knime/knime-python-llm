import knime.extension as knext
from ..base import model_category

ibm_watsonx_icon = "icons/ibm-bee.png"
ibm_watsonx_category = knext.category(
    path=model_category,
    name="IBM watsonx",
    level_id="ibm",
    description="IBM watsonx.ai models",
    icon=ibm_watsonx_icon,
)
