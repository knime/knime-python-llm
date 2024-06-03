# KNIME / own imports
import knime.extension as knext
import util

# Langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Other imports
import pyarrow as pa
import pandas as pd

util_icon = "icons/ml.png"


class OutputColumnSetting(knext.EnumParameterOptions):
    REPLACE = (
        "Replace",
        "The text chunks will replace the original texts.",
    )
    APPEND = (
        "Append",
        "The text chunks will be appended to the table in a new column.",
    )


# == Nodes ==


@knext.node(
    "Text Chunker (Labs)",
    knext.NodeType.MANIPULATOR,
    util_icon,
    category=util.main_category,
    keywords=[
        "Text Splitting",
        "GenAI",
        "RAG",
        "Retrieval Augmented Generation",
    ],
)
@knext.input_table("Input Table", "Table containing a string column.")
@knext.output_table("Result table", "Table containing the text chunks.")
class TextChunker:
    """
    Splits large texts into smaller overlapping chunks.

    Text chunking is a technique for splitting larger documents into smaller paragraphs. The chunks overlap
    to contain a piece of the context. Chunk size and overlap can be configured.
    """

    input_col = knext.ColumnParameter(
        label="Document column",
        description="Select the column containing the documents to be chunked.",
        port_index=0,
        column_filter=lambda col: col.ktype == knext.string(),
        include_none_column=False,
    )

    chunk_size = knext.IntParameter(
        "Chunk size",
        "Specify the maximum chunk size.",
        4000,
        min_value=1,
    )

    chunk_overlap = knext.IntParameter(
        "Chunk overlap",
        "Specify by how many characters the chunks should overlap.",
        200,
        min_value=0,
    )

    output_column = knext.EnumParameter(
        label="Output column",
        description="Select whether the chunks should replace the original column or be appended "
        "to the table in a new column.",
        default_value=OutputColumnSetting.REPLACE.name,
        enum=OutputColumnSetting,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    output_name = knext.StringParameter(
        "Output column name",
        "Provide the name of the new column containing the chunks.",
        "Chunk",
    ).rule(
        knext.OneOf(output_column, [OutputColumnSetting.APPEND.name]),
        knext.Effect.SHOW,
    )

    def configure(
        self,
        ctx: knext.ConfigurationContext,
        table_spec: knext.Schema,
    ) -> knext.Schema:
        if not self.input_col:
            self.input_col = util.pick_default_column(table_spec, knext.string())

        if self.chunk_overlap > self.chunk_size:
            raise knext.InvalidParametersError(
                "The chunk overlap must not be larger than the chunk size."
            )

        if self.output_column == OutputColumnSetting.APPEND.name:
            if not self.output_name:
                raise knext.InvalidParametersError(
                    "The output column name must not be empty."
                )

            table_spec = table_spec.append(
                knext.Column(
                    knext.list_(knext.string()),
                    util.handle_column_name_collision(
                        table_spec.column_names, self.output_name
                    ),
                )
            )
        else:
            idx = table_spec.column_names.index(self.input_col)
            table_spec = table_spec.remove(idx)
            table_spec = table_spec.insert(
                knext.Column(knext.list_(knext.string()), self.input_col), idx
            )

        return table_spec

    def execute(
        self,
        ctx: knext.ExecutionContext,
        input_table: knext.Table,
    ) -> knext.Table:
        pa_table = input_table.to_pyarrow()

        # Rename row ID column to avoid error when <RowID> column exists and is selected
        rowID_name = util.handle_column_name_collision(
            input_table.schema.column_names, "<RowID>"
        )
        pa_table = pa_table.rename_columns([rowID_name] + pa_table.column_names[1:])

        pa_col = pa_table.column(self.input_col)
        df = pa_col.to_pandas()

        # Apply Text Splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        df = df.apply(lambda row: splitter.split_text(row) if pd.notnull(row) else None)

        if self.output_column == OutputColumnSetting.APPEND.name:
            output_table = pa_table.append_column(
                util.handle_column_name_collision(
                    input_table.schema.column_names, self.output_name
                ),
                pa.Array.from_pandas(df, type=pa.list_(pa.string())),
            )
        else:
            index = pa_table.column_names.index(self.input_col)
            new_field = pa.field(self.input_col, pa.list_(pa.string()))
            new_column = pa.Array.from_pandas(df, type=pa.list_(pa.string()))
            output_table = pa_table.set_column(index, new_field, new_column)

        return knext.Table.from_pyarrow(output_table, row_ids="keep")
