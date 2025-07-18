# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
#  Copyright by KNIME AG, Zurich, Switzerland
#  Website: http://www.knime.com; Email: contact@knime.com
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License, Version 3, as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses>.
#
#  Additional permission under GNU GPL version 3 section 7:
#
#  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
#  Hence, KNIME and ECLIPSE are both independent programs and are not
#  derived from each other. Should, however, the interpretation of the
#  GNU GPL Version 3 ("License") under any applicable laws result in
#  KNIME and ECLIPSE being a combined program, KNIME AG herewith grants
#  you the additional permission to use and propagate KNIME together with
#  ECLIPSE with only the license terms in place for ECLIPSE applying to
#  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
#  license terms of ECLIPSE themselves allow for the respective use and
#  propagation of ECLIPSE together with KNIME.
#
#  Additional permission relating to nodes for KNIME that extend the Node
#  Extension (and in particular that are based on subclasses of NodeModel,
#  NodeDialog, and NodeView) and that only interoperate with KNIME through
#  standard APIs ("Nodes"):
#  Nodes are deemed to be separate and independent programs and to not be
#  covered works.  Notwithstanding anything to the contrary in the
#  License, the License does not apply to Nodes, you are not required to
#  license Nodes under the License, and you are granted a license to
#  prepare and propagate Nodes, in each case even if such Nodes are
#  propagated with or for interoperation with KNIME.  The owner of a Node
#  may freely choose the license terms applicable to such Node, including
#  when such Node is propagated with or for interoperation with KNIME.
# ------------------------------------------------------------------------


import knime.extension as knext
import util
from models.base import (
    LLMPortObject,
    LLMPortObjectSpec,
    llm_port_type,
)

import numpy as np
from typing import Optional
from json.decoder import JSONDecodeError

from ._base import (
    tortoise_icon,
    eval_category,
    _get_workflow_schema,
    _get_schema_from_workflow_spec,
    ScannerColumn,
    _validate_prediction_workflow_spec,
    _pick_default_workflow_column,
)


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
    "LLM",
    "The large language model used to analyze the workflow.",
    llm_port_type,
)
@knext.input_port(
    "Generative Workflow",
    "The generative workflow to analyze with Giskard.",
    knext.PortType.WORKFLOW,
)
@knext.input_table(
    "Dataset",
    "Dataset that is used to enhance the LLM-assisted detectors.",
    optional=True,
)
@knext.output_table("Giskard report data", "The Giskard scan report as table.")
@knext.output_view("Giskard report", "The Giskard scan report as HTML.")
class GiskardLLMScanner:
    """
    Evaluate the performance of GenAI workflows with Giskard.

    This node provides an open-source framework for detecting potential vulnerabilites arising from GenAI models in the provided workflow.
    It evaluates the workflow by combining heuristics-based and LLM-assisted detectors.
    Giskard uses the provided LLM for the evaluation but applies different model parameters for some of the detectors.
    The viability of the LLM-assisted detectors can be improved by providing an optional
    input table with common example prompts for the workflow.

    The node uses detectors for the following vulnerabilities:

    - *Hallucination and Misinformation*: Detects if the workflow is prone to generate fabricated or false information.
    - *Harmful Content*: Detects if the workflow is prone to produce content that is unethical, illegal or otherwise harmful.
    - *Prompt Injection*: Detects if the workflow's behavior can be altered via a variety of prompt injection techniques.
    - *Robustness*: Detects if the workflow is sensitive to small perturbations in the input that result in inconsistent responses.
    - *Stereotypes*: Detects stereotype-based discrimination in the workflow responses.
    - *Information Disclosure*: Attempts to cause the workflow to disclose sensitive information such as
    secrets or personally identifiable information. Might produce false-positives if the workflow is required to output information
    that can be considered sensitive such as contact information for a business.
    - *Output Formatting*: Checks that the workflow output is consistent with the format requirements indicated in the model description,
    if such instructions are provided.

    This node does not utilize Giskard's LLMCharsInjectionDetector.
    For more details on LLM vulnerabilities, refer to the
    [Giskard documentation](https://docs.giskard.ai/en/stable/knowledge/llm_vulnerabilities/index.html)

    In order to perform tasks with LLM-assisted detectors, Giskard sends the following information to the
    language model provider:

    - Data provided in your dataset
    - Text generated by your model
    - Model name and description

    Note that this does not apply if a self-hosted model is used.

    More information on Giskard can be found in the
    [documentation](https://docs.giskard.ai/en/stable/open_source/scan/scan_llm/index.html).

    **Note**: If you use the
    [Credentials Configuration node](https://hub.knime.com/knime/extensions/org.knime.features.js.quickforms/latest/org.knime.js.base.node.configuration.input.credentials.CredentialsDialogNodeFactory)
    and do not select the "Save password in configuration (weakly encrypted)" option for passing the API key for the LLM Selector node,
    the Credentials Configuration node will need to be reconfigured upon reopening the workflow, as the credentials flow variable
    was not saved and will therefore not be available to downstream nodes.
    """

    model_name = knext.StringParameter(
        "Workflow name",
        "A descriptive name for the workflow under evaluation. Used to generate domain-specific probes and included in the generated report.",
        "Workflow",
    )

    model_description = knext.StringParameter(
        "Workflow decription",
        "A more detailed description of the evaluated workflow that explains its purpose. Used to generate domain-specific probes.",
        "Workflow description",
    )

    feature_columns = knext.ColumnFilterParameter(
        "Feature columns",
        "The columns used as features by the generative workflow. Feature columns "
        "must be of type string. These columns must exist in the dataset if one is provided.",
        schema_provider=lambda ctx: _get_workflow_schema(ctx, 1, True),
        column_filter=lambda column: column.ktype == knext.string(),
    )

    response_column = knext.ColumnParameter(
        "Response column",
        "The column in the output table of the workflow that represents the LLM responses.",
        schema_provider=lambda ctx: _get_workflow_schema(ctx, 1, False),
        column_filter=lambda column: column.ktype == knext.string(),
    )

    ignore_errors = knext.BoolParameter(
        "Ignore detection errors",
        "If checked, execution will not stop when detection errors are encountered. Failed detectors will "
        "be ignored when creating the result.",
        False,
        is_advanced=True,
    )

    @property
    def output_columns(self):
        """Not a top-level constant to avoid importing pandas when importing the module."""
        import pandas as pd

        return [
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
        dataset_spec: Optional[knext.Schema],
    ) -> knext.Schema:
        llm_spec.validate_context(ctx)

        _validate_prediction_workflow_spec(prediction_workflow_spec)
        if not self.response_column:
            self.response_column = _pick_default_workflow_column(
                prediction_workflow_spec, False
            )

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
        dataset: Optional[knext.Table],
    ):
        import giskard as gk
        from giskard.llm.errors import LLMGenerationError
        from giskard.llm.client import set_default_client
        from ._llm_client import KnimeLLMClient
        import pandas as pd

        set_default_client(KnimeLLMClient(llm_port, ctx))

        workflow_table_spec = _get_schema_from_workflow_spec(workflow.spec, True)
        feature_names = self.feature_columns.apply(workflow_table_spec).column_names

        prompt_df = self._create_giskard_compatible_df(dataset, workflow_table_spec)

        giskard_dataset = gk.Dataset(
            df=prompt_df,
            target=None,
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
        # The unicode character is not displayed on some Windows machines
        html_report = html_report.replace("\xa0", "&nbsp;")

        df = self._catch_empty_dataframe(df)

        return (
            knext.Table.from_pandas(df),
            knext.view_html(html=html_report),
        )

    def _create_giskard_compatible_df(
        self, dataset: Optional[knext.Table], workflow_table_spec
    ):
        if dataset:
            dataset_df = dataset.to_pandas()
            # giskard expects string columns to be of type object
            for col in dataset.schema:
                if col.ktype == knext.string():
                    dataset_df[col.name] = dataset_df[col.name].astype("object")
        else:
            import pandas as pd

            dataset_df = pd.DataFrame(
                {
                    col: pd.Series(dtype="object")
                    for col in self.feature_columns.apply(
                        workflow_table_spec
                    ).column_names
                }
            )
        return dataset_df

    def _validate_feature_columns(
        self, workflow_spec, dataset_spec: Optional[knext.Schema]
    ) -> None:
        """Checks if the feature columns exist in the workflow input table and in the optional dataset table."""
        prediction_workflow_table = _get_schema_from_workflow_spec(
            workflow_spec, return_input_schema=True
        )

        feature_names = set(
            self.feature_columns.apply(prediction_workflow_table).column_names
        )

        if dataset_spec:
            if not feature_names.issubset(dataset_spec.column_names):
                raise knext.InvalidParametersError(
                    "Selected feature columns have to be in the dataset table."
                )

    def _validate_selected_params(
        self, workflow_spec, dataset_spec: Optional[knext.Schema]
    ) -> None:
        self._validate_feature_columns(workflow_spec, dataset_spec)

        if self.model_name == "":
            raise knext.InvalidParametersError(
                "The model name must not be empty, as it is used to create domain-specific probes."
            )
        if self.model_description == "":
            raise knext.InvalidParametersError(
                "The model description must not be empty, as it is used to create domain-specific probes."
            )
        if not self.feature_columns.apply(
            _get_schema_from_workflow_spec(workflow_spec, return_input_schema=True)
        ).column_names:
            raise knext.InvalidParametersError(
                "At least one feature column must be specified."
            )
        if not self.response_column:
            raise knext.InvalidParametersError("The response column must be specified.")

        util.check_column(
            _get_schema_from_workflow_spec(workflow_spec, return_input_schema=False),
            self.response_column,
            knext.string(),
            "response",
            "workflow input table",
        )

    def _enforce_string_data_types(self, df):
        for column in df.columns:
            df[column] = df[column].astype(str)

        return df

    def _catch_empty_dataframe(self, df):
        import pandas as pd

        if len(df.columns) == 0:
            df = pd.DataFrame(
                {col.name: pd.Series(dtype=col.pd_type) for col in self.output_columns}
            )
        return df
