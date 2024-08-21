import knime.extension as knext
import util
from models.base import (
    ChatModelPortObjectSpec,
    ChatModelPortObject,
    chat_model_port_type,
    EmbeddingsPortObjectSpec,
    EmbeddingsPortObject,
    embeddings_model_port_type,
)

from ._base import tortoise_icon, KnimeLLMClient

import pandas as pd
import numpy as np

from giskard.llm.client import set_default_client
from giskard.rag.testset_generation import generate_testset, KnowledgeBase


class _KnowledgeBase(KnowledgeBase):
    """
    A custom subclass of KnowledgeBase from giskard.rag that overrides the _embeddings property.

    Instead of computing embeddings, this class utilizes precomputed embeddings
    provided by the vector store.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        documents_col: str,
        embeddings_col: str,
        embeddings_model,
    ):
        super().__init__(
            data=data, columns=[documents_col], embedding_model=embeddings_model
        )
        # initialize embeddings so they don't get recalculated in the _embeddings property
        vectors = data[embeddings_col]
        self._embeddings_inst = np.array([np.array(vec) for vec in vectors])
        for doc, emb in zip(self._documents, self._embeddings_inst):
            doc.embeddings = emb


@knext.parameter_group("Data")
class InputDataParameters:
    documents_col = knext.ColumnParameter(
        "Documents",
        "The column containing the documents in the knowledge base.",
        port_index=2,
        column_filter=lambda c: c.ktype == knext.string(),
    )

    embeddings_col = knext.ColumnParameter(
        "Embeddings",
        "The column containing the embeddings in the knowledge base.",
        port_index=2,
        column_filter=lambda c: c.ktype == knext.ListType(knext.double()),
    )


@knext.parameter_group("Test Set")
class TestSetParameters:
    description = knext.StringParameter(
        "Agent Description",
        """A brief description of the agent to be evaluated. This information will be 
        used to generate appropriate test questions tailored to the system.""",
    )

    n_questions = knext.IntParameter(
        "Number of Questions",
        "Specifies the number of test questions to generate.",
        10,
        min_value=1,
    )


@knext.node(
    "Giskard RAGET Test Set Generator",
    knext.NodeType.OTHER,
    tortoise_icon,
    category=util.main_category,
    keywords=[
        "GenAI",
        "Gen AI",
        "Generative AI",
        "Model evaluation",
        "Text generation",
        "Large Language Model",
        "Chat Model",
        "Retrieval Augmented Generation",
    ],
)
@knext.input_port(
    "Chat Model",
    """A configured Chat Model used to generate question from the provided knowledge base.""",
    chat_model_port_type,
)
@knext.input_port(
    "Embedding Model",
    """A configured Embedding Model used to embed queries to find related documents.""",
    embeddings_model_port_type,
)
@knext.input_table(
    "Knowledge Base",
    """A table containing documents and their embedded representations that are utilized by the RAG system.""",
)
@knext.output_table("Test Set", "The generated test set.")
class TestSetGenerator:
    """
    Generates a test set for evaluating a RAG system.

    This node utilizes the provided knowledge base and the task description of the RAG agent
    to automatically generate a diverse set of test questions, reference answers, and reference
    contexts. The generated test set is designed to assess the performance of various components
    of the RAG system, such as the retriever, generator, and knowledge base quality. The
    questions target specific aspects of the RAG system, helping to identify potential
    weaknesses and areas for improvement.
    """

    input_data = InputDataParameters()
    test_set_params = TestSetParameters()

    def configure(
        self,
        ctx,
        chat_model_spec: ChatModelPortObjectSpec,
        embed_model_spec: EmbeddingsPortObjectSpec,
        table_spec: knext.Schema,
    ):
        chat_model_spec.validate_context(ctx)
        embed_model_spec.validate_context(ctx)

        self._check_column_types_exist(table_spec)
        self._set_default_columns(table_spec)

        return knext.Schema.from_columns(
            [
                knext.Column(knext.string(), "id"),
                knext.Column(knext.string(), "question"),
                knext.Column(knext.string(), "reference_answer"),
                knext.Column(knext.string(), "reference_context"),
                knext.Column(knext.logical(dict), "conversation_history"),
                knext.Column(knext.logical(dict), "metadata"),
            ]
        )

    def execute(
        self,
        ctx,
        chat_model_port_object: ChatModelPortObject,
        embed_model_port_object: EmbeddingsPortObject,
        input_table: knext.Table,
    ):
        set_default_client(KnimeLLMClient(chat_model_port_object, ctx))

        df = input_table.to_pandas()

        kb = _KnowledgeBase(
            data=df,
            documents_col=self.input_data.documents_col,
            embeddings_col=self.input_data.embeddings_col,
            embeddings_model=embed_model_port_object.create_model(ctx),
        )

        testset = generate_testset(
            knowledge_base=kb,
            num_questions=self.test_set_params.n_questions,
            agent_description=self.test_set_params.description,
        ).to_pandas()

        testset["conversation_history"] = testset["conversation_history"].apply(
            lambda row: {"conversation_history": row}
        )

        testset = testset.reset_index()
        testset = testset.rename(columns={"ID": "id"})

        return knext.Table.from_pandas(testset)

    def _check_column_types_exist(self, table_spec):
        has_string_column = False
        has_vector_column = False

        for col in table_spec:
            if col.ktype == knext.string():
                has_string_column = True
            elif col.ktype == knext.ListType(knext.double()):
                has_vector_column = True

            if has_string_column and has_vector_column:
                break

        if not has_string_column:
            raise knext.InvalidParametersError(
                "The knowledge base must contain at least one string column."
            )

        if not has_vector_column:
            raise knext.InvalidParametersError(
                "The knowledge base must contain at least one vector column. A list of doubles is expected."
            )

    def _set_default_columns(self, table_spec):
        if not self.input_data.documents_col:
            self.input_data.documents_col = util.pick_default_column(
                table_spec, knext.string()
            )

        if not self.input_data.embeddings_col:
            self.input_data.embeddings_col = util.pick_default_column(
                table_spec, knext.ListType(knext.double())
            )
