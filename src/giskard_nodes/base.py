import knime.extension as knext
import util
from models.base import LLMPortObject, LLMPortObjectSpec, llm_port_type

import os

os.environ["GSK_DISABLE_SENTRY"] = "True"

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Optional
from json.decoder import JSONDecodeError
from langchain_core.language_models import BaseLanguageModel, BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, ChatMessage, SystemMessage

import giskard as gk
from giskard.llm.client import set_default_client
from giskard.llm.client.base import LLMClient
from giskard.llm.errors import LLMGenerationError


tortoise_icon = "icons/tortoise_icon.png"


eval_category = knext.category(
    path=util.main_category,
    level_id="evaluation",
    name="Evaluation",
    description="",
    icon=tortoise_icon,
)


def _get_schema_from_workflow_spec(
    workflow_spec, return_input_schema: bool
) -> knext.Schema:
    if workflow_spec is None:
        raise ValueError("Workflow spec is not available. Execute predecessor nodes.")
    if return_input_schema:
        return next(iter(workflow_spec.inputs.values())).schema
    else:
        return next(iter(workflow_spec.outputs.values())).schema


def _get_workflow_schema(ctx: knext.DialogCreationContext) -> knext.Schema:
    return _get_schema_from_workflow_spec(ctx.get_input_specs()[1], False)


def _check_workflow_column(
    input_table: knext.Schema,
    column_name: str,
    expected_type: knext.KnimeType,
    column_purpose: str,
) -> None:
    """
    Raises an InvalidParametersError if a column named column_name is not contained in the table of a workflow or has the wrong KnimeType.
    """
    if column_name not in input_table.column_names:
        raise knext.InvalidParametersError(
            f"The {column_purpose} column '{column_name}' is missing in the prediction workflow table."
        )
    ktype = input_table[column_name].ktype
    if ktype != expected_type:
        raise knext.InvalidParametersError(
            f"The {column_purpose} column '{column_name}' is of type {str(ktype)} but should be of type {str(expected_type)}."
        )


class KnimeLLMClient(LLMClient):
    def __init__(self, model: BaseLanguageModel):
        self._model = model

    def complete(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        caller_id: Optional[str] = None,
        seed: Optional[int] = None,
        format=None,
    ) -> ChatMessage:
        """Prompts the model to generate domain-specific probes. Uses the parameters of the model instead of
        the function parameters."""

        converted_messages = self._convert_messages(messages)
        answer = self._model.invoke(converted_messages)
        if isinstance(self._model, BaseChatModel):
            answer = answer.content
        return ChatMessage(role="assistant", content=answer)

    _role_to_message_type = {
        "ai": AIMessage,
        "assistant": AIMessage,
        "user": HumanMessage,
        "human": HumanMessage,
        "system": SystemMessage,
    }

    def _create_message(self, role: str, content: str):
        if not role:
            raise RuntimeError("Giskard did not specify a message role.")
        message_type = self._role_to_message_type.get(role.lower(), None)
        if message_type:
            return message_type(content=content)
        else:
            # fallback
            return ChatMessage(content=content, role=role)

    def _convert_messages(self, messages: Sequence[ChatMessage]):
        return [self._create_message(msg.role, msg.content) for msg in messages]


@dataclass
class ScannerColumn:
    name: str
    knime_type: knext.KnimeType
    pd_type: type


# == Nodes ==


@knext.node(
    "Giskard LLM Scanner",
    knext.NodeType.OTHER,
    tortoise_icon,
    category=eval_category,
    keywords=[
        "Model evaluation",
        "Machine learning",
        "Text generation",
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Large Language Model",
    ],
)
@knext.input_port(
    "LLM or chat model",
    "The large language model or chat model used to analyze the workflow.",
    llm_port_type,
)
@knext.input_port(
    "Prediction workflow",
    "The prediction workflow to analyze with Giskard.",
    knext.PortType.WORKFLOW,
)
@knext.input_table(
    "Dataset",
    "Dataset containing prompts that are used to enhance the LLM-assisted detectors.",
)
@knext.output_table("Giskard report data", "The Giskard scan report as table.")
@knext.output_view("Giskard report", "The Giskard scan report as HTML.")
class GiskardLLMScanner:
    """
    Evaluate LLM Model Performance with Giskard.

    This node provides an open-source framework for detecting potential vulnerabilites in LLM models provided
    as a workflow. It evaluates LLM models by combining heuristics-based and LLM-assisted detectors.

    In order to perform tasks with LLM-assisted detectors, Giskard sends the following information to the
    language model provider:

    - Data provided in your Dataset
    - Text generated by your model
    - Model name and description

    Note that this does not apply if a self-hosted model is used. This node does not utilize Giskard's
    LLMCharsInjectionDetector. More information on Giskard can be found in the
    [documentation](https://docs.giskard.ai/en/stable/open_source/scan/scan_llm/index.html).
    """

    dataset_name = knext.StringParameter(
        "Dataset name",
        "The name of the dataset. Used in the generated report.",
        "dataset",
    )

    model_name = knext.StringParameter(
        "Model name",
        "The model name. Used to generate domain-specific probes and included in the generated report.",
        "model",
    )

    model_description = knext.StringParameter(
        "Model decription",
        "The model description. Used to generate domain-specific probes.",
        "model description",
    )

    prompt_column = knext.ColumnParameter(
        "Prompt column",
        "The column of your dataset that contains the prompts for the model.",
        port_index=2,
        column_filter=lambda column: column.ktype == knext.string(),
    )

    response_column = knext.ColumnParameter(
        "Response column",
        "The column in the output table of the workflow that represents the LLM responses.",
        schema_provider=_get_workflow_schema,
        column_filter=lambda column: column.ktype == knext.string(),
    )

    ignore_errors = knext.BoolParameter(
        "Ignore detection errors",
        "If checked the execution will not stop when detection errors are encountered. Failed detectors will "
        "be ignored when creating the result.",
        False,
        is_advanced=True,
    )

    output_columns = [
        ScannerColumn("domain", knext.string(), pd.StringDtype()),
        ScannerColumn("slicing_fn", knext.string(), pd.StringDtype()),
        ScannerColumn("transformation_fn", knext.string(), pd.StringDtype()),
        ScannerColumn("metric", knext.string(), pd.StringDtype()),
        ScannerColumn("deviation", knext.string(), pd.StringDtype()),
        ScannerColumn("description", knext.string(), pd.StringDtype()),
    ]

    def configure(
        self,
        ctx,
        llm_spec: LLMPortObjectSpec,
        prediction_workflow_spec,
        dataset_spec: knext.Schema,
    ) -> knext.Schema:
        llm_spec.validate_context(ctx)

        self._validate_prediction_workflow_spec(prediction_workflow_spec)
        self._validate_prompt_column(prediction_workflow_spec, dataset_spec)
        self._pick_default_response_column(prediction_workflow_spec)
        self._validate_selected_params(prediction_workflow_spec, dataset_spec)

        return knext.Schema.from_columns(
            [
                knext.Column(
                    col.knime_type,
                    col.name,
                )
                for col in self.output_columns
            ]
        )

    def execute(
        self,
        ctx: knext.ExecutionContext,
        llm_port: LLMPortObject,
        workflow,
        dataset: knext.Table,
    ):
        llm = llm_port.create_model(ctx)
        set_default_client(KnimeLLMClient(llm))

        feature_names = [self.prompt_column]

        prompt_df = self._create_giskard_compatible_df(dataset)

        giskard_dataset = gk.Dataset(
            df=prompt_df,
            target=None,
            name=self.dataset_name,
        )

        input_key = next(iter(workflow.spec.inputs))

        # defined here to be accessed by the prediction_function for progress reporting
        detectors = []
        current_detector = 0

        def prediction_function(df: pd.DataFrame) -> np.ndarray:
            if ctx.is_canceled():
                raise RuntimeError("Execution canceled.")
            nonlocal current_detector
            nonlocal detectors
            detector = None
            if detectors and current_detector < len(detectors):
                detector = detectors[current_detector]
                ctx.set_progress(
                    current_detector / len(detectors),
                    f"Analyzing model with {type(detector).__name__}.",
                )
            table = knext.Table.from_pandas(df)
            outputs, _ = workflow.execute({input_key: table})
            predictions_df = outputs[0][self.response_column].to_pandas()

            if detector:
                current_detector = current_detector + 1
                ctx.set_progress(
                    current_detector / len(detectors),
                    f"Finished analyzing with {type(detector).__name__}.",
                )
            return predictions_df[self.response_column].tolist()

        giskard_model = gk.Model(
            model=prediction_function,
            model_type="text_generation",
            name=self.model_name,
            description=self.model_description,
            feature_names=feature_names,
        )

        # Remove CharsInjectionParameter to avoid torch import
        del gk.scanner.registry.DetectorRegistry._detectors["llm_chars_injection"]
        del gk.scanner.registry.DetectorRegistry._tags["llm_chars_injection"]

        scanner = gk.scanner.Scanner()

        # the detectors are used for progress reporting in the prediction_function
        detectors = list(
            scanner.get_detectors(tags=[giskard_model.meta.model_type.value])
        )

        try:
            # verbose is set to False because giskards print output is not supported for all system encodings
            # especially the Windows encoding cp1252
            scan_result = scanner.analyze(
                model=giskard_model,
                dataset=giskard_dataset,
                verbose=False,
                raise_exceptions=not self.ignore_errors,
            )
        except LLMGenerationError as exc:
            raise LLMGenerationError(
                "A detector failed. This could be because the maximum response length is not large enough or "
                "the response of the model could not be parsed to JSON. If this occurs often, consider using "
                "a different model for evaluation or enabling 'Ignore detection errors'."
            ) from exc
        except KeyError as exc:
            raise KeyError(
                "A detector failed because it could not find a feature column. This could be because the model "
                "did not respond with the correct column name when generating domain-specific probes. If this "
                "occurs often, consider using a different model for evaluation or enabling 'Ignore detection errors'."
            ) from exc
        except JSONDecodeError as exc:
            raise RuntimeError(
                "A detector failed because the response of the model could not be parsed to JSON. If this "
                "occurs often, consider using a different model for evaluation or enabling 'Ignore detection errors'."
            ) from exc
        except UnicodeEncodeError as exc:
            raise RuntimeError(
                "A detector failed because the model created a probe that included a unicode character that "
                "could not be encoded. If this occurs often, consider using a different model for evaluation or "
                "enabling 'Ignore detection errors'."
            ) from exc
        except ValueError as exc:
            raise ValueError(
                "A detector failed. This could be because the model did not include all features when "
                "generating domain-specific probes. If this occurs often, consider using a different model "
                "for evaluation or enabling 'Ignore detection errors'."
            ) from exc

        df = self._enforce_string_data_types(scan_result.to_dataframe())

        html_report = scan_result.to_html()

        df = self._catch_empty_dataframe(df)

        return (
            knext.Table.from_pandas(df),
            knext.view_html(html=html_report),
        )

    def _create_giskard_compatible_df(self, dataset: knext.Table):
        dataset_df = dataset.to_pandas()
        # giskard expects string columns to be of type object
        for col in dataset.schema:
            if col.ktype == knext.string():
                dataset_df[col.name] = dataset_df[col.name].astype("object")
        return dataset_df

    def _validate_prediction_workflow_spec(self, workflow_spec) -> None:
        if len(workflow_spec.inputs) != 1:
            raise knext.InvalidParametersError(
                "Prediction workflow must have exactly one input table."
            )

        if len(workflow_spec.outputs) != 1:
            raise knext.InvalidParametersError(
                "Prediction workflow must produce exactly one output table."
            )

    def _validate_prompt_column(
        self, workflow_spec, dataset_spec: knext.Schema
    ) -> None:
        """Validates whether the prompt column exists in the input of the workflow. Selects a default if no
        column is specified."""
        if not self.prompt_column:
            self.prompt_column = util.pick_default_column(dataset_spec, knext.string())

        prediction_workflow_input_cols = _get_schema_from_workflow_spec(
            workflow_spec, return_input_schema=True
        ).column_names

        if self.prompt_column not in prediction_workflow_input_cols:
            raise knext.InvalidParametersError(
                "Selected prompt column has to be in the input table of the prediction workflow."
            )

    def _validate_selected_params(
        self, workflow_spec, dataset_spec: knext.Schema
    ) -> None:
        if self.dataset_name == "":
            raise knext.InvalidParametersError("The dataset name must not be empty.")
        if self.model_name == "":
            raise knext.InvalidParametersError(
                "The model name must not be empty, as it is used to create domain-specific probes."
            )
        if self.model_description == "":
            raise knext.InvalidParametersError(
                "The model description must not be empty, as it is used to create domain-specific probes."
            )
        if not self.prompt_column:
            raise knext.InvalidParametersError("The prompt column must be specified.")
        if not self.response_column:
            raise knext.InvalidParametersError("The response column must be specified.")

        util.check_column(dataset_spec, self.prompt_column, knext.string(), "prompt")
        _check_workflow_column(
            _get_schema_from_workflow_spec(workflow_spec, return_input_schema=False),
            self.response_column,
            knext.string(),
            "response",
        )

    def _pick_default_response_column(self, workflow_spec) -> None:
        if not self.response_column:
            prediction_workflow_output_schema = _get_schema_from_workflow_spec(
                workflow_spec, return_input_schema=False
            )
            self.response_column = util.pick_default_column(
                prediction_workflow_output_schema, knext.string()
            )

    def _enforce_string_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in df.columns:
            df[column] = df[column].astype(str)

        return df

    def _catch_empty_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df.columns) == 0:
            df = pd.DataFrame(
                {col.name: pd.Series(dtype=col.pd_type) for col in self.output_columns}
            )
        return df
