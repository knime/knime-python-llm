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


# KNIME / own imports
import knime.extension as knext
from .base import (
    LLMPortObjectSpec,
    LLMPortObject,
    ChatModelPortObjectSpec,
    ChatModelPortObject,
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    OutputFormatOptions,
    model_category,
    util,
)

# Other imports
import pickle

fake_icon = "icons/fake.png"
test_category = knext.category(
    path=model_category,
    level_id="test",
    name="Testing",
    description="",
    icon=fake_icon,
)
# == Settings ==


class MissingValueHandlingOptions(knext.EnumParameterOptions):
    DefaultAnswer = (
        "Default response",
        "Return a predefined response.",
    )
    Fail = (
        "Fail",
        "Fail if no matching query is found.",
    )


@knext.parameter_group(label="Test Configuration")
class TestNodesGeneralSettings:
    test_prompt_col = knext.ColumnParameter(
        "Prompt column",
        "Column containing the test prompts.",
        port_index=0,
        column_filter=util.create_type_filter(knext.string()),
    )

    test_response_col = knext.ColumnParameter(
        "Response column",
        "Column containing the test responses for each prompt.",
        port_index=0,
        column_filter=util.create_type_filter(knext.string()),
    )


@knext.parameter_group(label="Prompt mismatch")
class MismatchSettings:
    prompt_mismatch = knext.EnumParameter(
        "Handle prompt mismatch",
        """Specify whether the LLM should provide a default response or fail when a given prompt cannot 
        be found in the prompt column.""",
        default_value=lambda v: MissingValueHandlingOptions.DefaultAnswer.name,
        enum=MissingValueHandlingOptions,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    default_response = knext.StringParameter(
        "Default response",
        "The test LLM's response when a query is not found in the prompt column.",
        "Could not find an answer to the query.",
    ).rule(
        knext.OneOf(prompt_mismatch, [MissingValueHandlingOptions.DefaultAnswer.name]),
        knext.Effect.SHOW,
    )


@knext.parameter_group(label="Embeddings Model Configuration")
class TestEmbeddingsSettings:
    test_doc_col = knext.ColumnParameter(
        "Document column",
        "Column containing the test documents as strings.",
        port_index=0,
        column_filter=util.create_type_filter(knext.string()),
    )

    test_vector_col = knext.ColumnParameter(
        "Vector column",
        """Column containing the test vectors. The column is expected to be a list of doubles. You may create the list 
        using the 
        [Table Creator](https://hub.knime.com/knime/extensions/org.knime.features.base/latest/org.knime.base.node.io.tablecreator.TableCreator2NodeFactory) and
        the [Create Collection Column](https://hub.knime.com/knime/extensions/org.knime.features.base/latest/org.knime.base.collection.list.create2.CollectionCreate2NodeFactoryhttps://hub.knime.com/knime/extensions/org.knime.features.base/latest/org.knime.base.collection.list.create2.CollectionCreate2NodeFactory) 
        nodes.""",
        port_index=0,
        column_filter=util.create_type_filter(knext.ListType(knext.double())),
    )

    fail_on_mismatch = knext.BoolParameter(
        "Fail on retrieval mismatch",
        "Whether the Test Embeddings Model should fail downstream on document mismatch.",
        default_value=False,
    )


# == Fake Implementations for testing Nodes ==


def _warn_if_same_columns(ctx, f_prompt_col: str, response_col: str) -> None:
    if f_prompt_col == response_col:
        ctx.set_warning("Query and response column are set to be the same column.")


# == Port Objects ==


class TestLLMPortObjectSpec(LLMPortObjectSpec):
    def __init__(self, missing_value_strategy: str) -> None:
        super().__init__()
        self._missing_value_strategy = missing_value_strategy

    @property
    def missing_value_strategy(self) -> str:
        return self._missing_value_strategy

    def serialize(self) -> dict:
        return {
            "missing_value_strategy": self._missing_value_strategy,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return TestLLMPortObjectSpec(data["missing_value_strategy"])


class TestLLMPortObject(LLMPortObject):
    def __init__(
        self,
        spec: TestLLMPortObjectSpec,
        responses: dict[str, str],
        default_response: str,
    ) -> None:
        super().__init__(spec)
        self._responses = responses
        self._default_response = default_response

    @property
    def spec(self) -> TestLLMPortObjectSpec:
        return super().spec

    @property
    def responses(self):
        return self._responses

    @property
    def default_response(self):
        return self._default_response

    def serialize(self):
        return pickle.dumps((self._responses, self._default_response))

    @classmethod
    def deserialize(cls, spec, data):
        (responses, default_response) = pickle.loads(data)
        return cls(spec, responses, default_response)

    def create_model(
        self, ctx: knext.ExecutionContext, output_format: OutputFormatOptions
    ):
        from ._fake_models import TestDictLLM

        return TestDictLLM(
            response_dict=self.responses,
            default_response=self.default_response,
            missing_value_strategy=self.spec.missing_value_strategy,
        )


test_llm_port_type = knext.port_type(
    "Test LLM", TestLLMPortObject, TestLLMPortObjectSpec
)


class TestChatModelPortObjectSpec(ChatModelPortObjectSpec):
    def __init__(self, missing_value_strategy: str) -> None:
        super().__init__()
        self._missing_value_strategy = missing_value_strategy

    @property
    def missing_value_strategy(self) -> str:
        return self._missing_value_strategy

    def serialize(self) -> dict:
        return {
            "missing_value_strategy": self._missing_value_strategy,
        }

    @classmethod
    def deserialize(cls, data: dict):
        return TestLLMPortObjectSpec(data["missing_value_strategy"])


class TestChatModelPortObject(ChatModelPortObject):
    def __init__(
        self,
        spec: TestLLMPortObjectSpec,
        responses: dict[str, str],
        default_response: str,
    ) -> None:
        super().__init__(spec)
        self._responses = responses
        self._default_response = default_response

    @property
    def spec(self) -> TestChatModelPortObjectSpec:
        return super().spec

    @property
    def responses(self):
        return self._responses

    @property
    def default_response(self):
        return self._default_response

    def serialize(self):
        return pickle.dumps((self._responses, self._default_response))

    @classmethod
    def deserialize(cls, spec, data):
        (responses, default_response) = pickle.loads(data)
        return cls(spec, responses, default_response)

    def create_model(
        self, ctx: knext.ExecutionContext, output_format: OutputFormatOptions
    ):
        from ._fake_models import TestChatModel

        return TestChatModel(
            response_dict=self.responses,
            default_response=self.default_response,
            missing_value_strategy=self.spec.missing_value_strategy,
        )


test_chat_model_port_type = knext.port_type(
    "Test Chat Model", TestChatModelPortObject, TestChatModelPortObjectSpec
)


class TestEmbeddingsPortObjectSpec(EmbeddingsPortObjectSpec):
    def __init__(self, fail_on_mismatch: bool) -> None:
        super().__init__()
        self._fail_on_mismatch = fail_on_mismatch

    @property
    def fail_on_mismatch(self) -> bool:
        return self._fail_on_mismatch

    def serialize(self) -> dict:
        return {"fail_on_mismatch": self._fail_on_mismatch}

    @classmethod
    def deserialize(cls, data: dict):
        return TestEmbeddingsPortObjectSpec(data["fail_on_mismatch"])


class TestEmbeddingsPortObject(EmbeddingsPortObject):
    def __init__(
        self,
        spec: TestEmbeddingsPortObjectSpec,
        embeddings_dict: dict[str, list[float]],
    ):
        super().__init__(spec)
        self._embeddings_dict = embeddings_dict

    @property
    def embeddings_dict(self):
        return self._embeddings_dict

    def serialize(self):
        return pickle.dumps(self._embeddings_dict)

    @classmethod
    def deserialize(cls, spec, data: dict):
        embeddings_dict = pickle.loads(data)
        return TestEmbeddingsPortObject(spec, embeddings_dict)

    def create_model(self, ctx):
        from ._fake_models import TestEmbeddings

        return TestEmbeddings(
            embeddings_dict=self.embeddings_dict,
            fail_on_mismatch=self.spec.fail_on_mismatch,
        )


test_embeddings_port_type = knext.port_type(
    "Test Embeddings", TestEmbeddingsPortObject, TestEmbeddingsPortObjectSpec
)


def _configure_model_generation_columns(
    table_spec: knext.Schema,
    test_prompt_col: str,
    test_response_col: str,
    response_includes_vectors: bool = None,
) -> tuple[str, str]:
    if test_prompt_col and test_response_col:
        util.check_column(
            table_spec,
            test_prompt_col,
            knext.string(),
            "test prompt",
        )

        if response_includes_vectors:
            util.check_column(
                table_spec,
                test_response_col,
                knext.ListType(knext.double()),
                "test embeddings",
            )
        else:
            util.check_column(
                table_spec,
                test_response_col,
                knext.string(),
                "test response",
            )

        return test_prompt_col, test_response_col

    else:
        if response_includes_vectors:
            query_col = util.pick_default_column(table_spec, knext.string())
            vector_col = util.pick_default_column(
                table_spec, knext.ListType(knext.double())
            )
            return query_col, vector_col

        default_columns = util.pick_default_columns(table_spec, knext.string(), 2)
        return default_columns[0], default_columns[1]


# == Nodes ==


def to_dictionary(table, columns: list[str]):
    df = table.to_pandas()

    for col in columns:
        is_missing = df[col].isnull().any()

        if is_missing:
            raise knext.InvalidParametersError(
                f"""Missing value found in column '{col}'. Please ensure, that the query 
                and response columns do not have missing values."""
            )

    response_dict = dict(
        map(
            lambda i, j: (i, j),
            df[columns[0]],
            df[columns[1]],
        )
    )

    return response_dict


@knext.node(
    "Test Instruct Model Selector",
    knext.NodeType.SOURCE,
    fake_icon,
    category=test_category,
    is_hidden=True,
)
@knext.input_table(
    "Prompt and response table",
    "A table with columns for test prompts and their corresponding responses.",
)
@knext.output_port(
    "Test LLM",
    "Configured test LLM",
    test_llm_port_type,
)
class TestLLMConnector:
    """
    Creates a Test Large Language Model.

    This node creates a Test Large Language Model (LLM) implementation for testing
    purposes without the need for heavy computing power. Provide a column with expected
    prompts and their respective answers in a second column. When the 'Test Model'
    is prompted with a matching prompt, it will return the provided answer. If the prompt
    does not match, it will return a default value or fail based on its configuration.
    """

    settings = TestNodesGeneralSettings()
    mismatch = MismatchSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        table_spec: knext.Schema,
    ) -> TestLLMPortObjectSpec:
        test_prompt_col, test_response_col = _configure_model_generation_columns(
            table_spec, self.settings.test_prompt_col, self.settings.test_response_col
        )

        _warn_if_same_columns(ctx, test_prompt_col, test_response_col)

        self.settings.test_prompt_col = test_prompt_col
        self.settings.test_response_col = test_response_col

        return self.create_spec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        input_table: knext.Table,
    ) -> TestLLMPortObject:
        _warn_if_same_columns(
            ctx, self.settings.test_prompt_col, self.settings.test_response_col
        )

        response_dict = to_dictionary(
            input_table,
            [self.settings.test_prompt_col, self.settings.test_response_col],
        )

        return TestLLMPortObject(
            self.create_spec(),
            response_dict,
            self.mismatch.default_response,
        )

    def create_spec(self) -> TestLLMPortObjectSpec:
        return TestLLMPortObjectSpec(self.mismatch.prompt_mismatch)


@knext.node(
    "Test Chat Model Selector",
    knext.NodeType.SOURCE,
    fake_icon,
    category=test_category,
    is_hidden=True,
)
@knext.input_table(
    "Prompt and response table",
    "A table with columns for test prompts and their corresponding responses.",
)
@knext.output_port(
    "Test Chat Model",
    "Configured Test Chat Model",
    test_chat_model_port_type,
)
class TestChatModelConnector:
    """
    Creates a Test Chat Model.

    This node creates a Test Chat Model implementation for testing purposes without the
    need for heavy computing power. Provide a column with prompts and their respective
    responses in a second column. When the 'Test Chat Model' is prompted,
    it will return the provided answer as an AI message. If the prompt
    does not match, it will return a default value or fail based on its configuration.
    """

    settings = TestNodesGeneralSettings()
    mismatch = MismatchSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        table_spec: knext.Schema,
    ) -> TestChatModelPortObjectSpec:
        test_prompt_col, test_response_col = _configure_model_generation_columns(
            table_spec, self.settings.test_prompt_col, self.settings.test_response_col
        )

        _warn_if_same_columns(ctx, test_prompt_col, test_response_col)

        self.settings.test_prompt_col = test_prompt_col
        self.settings.test_response_col = test_response_col

        return self.create_spec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        input_table: knext.Table,
    ) -> TestChatModelPortObject:
        df = input_table.to_pandas()

        _warn_if_same_columns(
            ctx, self.settings.test_prompt_col, self.settings.test_response_col
        )

        response_dict = dict(
            map(
                lambda i, j: (i, j),
                df[self.settings.test_prompt_col],
                df[self.settings.test_response_col],
            )
        )

        return TestChatModelPortObject(
            self.create_spec(),
            response_dict,
            self.mismatch.default_response,
        )

    def create_spec(self) -> TestChatModelPortObjectSpec:
        return TestChatModelPortObjectSpec(self.mismatch.prompt_mismatch)


@knext.node(
    "Test Embeddings Connector",
    knext.NodeType.SOURCE,
    fake_icon,
    category=test_category,
    is_hidden=True,
)
@knext.input_table(
    "Document and vector table",
    "A table containing a column with documents and one with the respective vectors.",
)
@knext.output_port(
    "Test Embeddings",
    "Configured Test Embeddings Model",
    test_embeddings_port_type,
)
class TestEmbeddingsConnector:
    """
    Creates a Test Embeddings Model.

    This node creates a Test Embeddings Model implementation for testing purposes without
    the need for heavy computing power. Provide a set of documents and queries along with their
    corresponding vectors for the following nodes, e.g., Vector Store Creators and Vector Store Retriever,
    to use.

    All downstream nodes working with the Test Embeddings Model need to be supplied with matching documents,
    which should also be used as queries in the Vector Store Retriever node.
    Failure to do so will, based on the configuration, either result in errors or return documents closest to the
    zero vector.

    With this node you simulate exactly with which vectors documents will be stored and retrieved.
    """

    settings = TestEmbeddingsSettings()

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        table_spec: knext.Schema,
    ) -> TestEmbeddingsPortObjectSpec:
        test_doc_col, test_vector_col = _configure_model_generation_columns(
            table_spec,
            self.settings.test_doc_col,
            self.settings.test_vector_col,
            response_includes_vectors=True,
        )

        _warn_if_same_columns(ctx, test_doc_col, test_vector_col)

        self.settings.test_doc_col = test_doc_col
        self.settings.test_vector_col = test_vector_col

        return self.create_spec()

    def execute(
        self,
        ctx: knext.ExecutionContext,
        input_table: knext.Table,
    ) -> TestEmbeddingsPortObject:
        df = input_table.to_pandas()

        embeddings_dict = dict(
            map(
                lambda i, j: (i, j.tolist()),
                df[self.settings.test_doc_col],
                df[self.settings.test_vector_col],
            )
        )

        return TestEmbeddingsPortObject(self.create_spec(), embeddings_dict)

    def create_spec(self) -> TestEmbeddingsPortObjectSpec:
        return TestEmbeddingsPortObjectSpec(self.settings.fail_on_mismatch)
