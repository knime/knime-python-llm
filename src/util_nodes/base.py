# KNIME / own imports
import knime.extension as knext
import util

# Langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Other imports
import pyarrow as pa
import pyarrow.compute as pc
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

            return table_spec.append(
                knext.Column(
                    knext.string(),
                    util.handle_column_name_collision(
                        table_spec.column_names, self.output_name
                    ),
                )
            )
        else:
            return table_spec

    def execute(
        self,
        ctx: knext.ExecutionContext,
        input_table: knext.Table,
    ) -> knext.Table:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        return knext.BatchOutputTable.from_batches(
            self._generate_batches(
                input_table,
                splitter,
            ),
            row_ids="keep",
        )

    def _generate_batches(
        self,
        input_table: knext.Table,
        splitter: RecursiveCharacterTextSplitter,
    ):
        for batch in input_table.batches():
            pa_batch = batch.to_pyarrow()
            pa_table = pa.Table.from_batches([pa_batch])

            append_column_field = pa.field(
                util.handle_column_name_collision(
                    input_table.schema.column_names, self.output_name
                ),
                pa.string(),
            )

            # Construct empty table if input table is empty to avoid error
            if pa_table.shape[0] == 0:
                if self.output_column == OutputColumnSetting.APPEND.name:
                    empty_array = pa.array([], type=pa.string())
                    pa_table = pa_table.append_column(append_column_field, empty_array)

                yield knext.Table.from_pyarrow(pa_table)
            else:
                # Rename row ID column to avoid error when <RowID> column exists and is selected
                rowID_name = util.handle_column_name_collision(
                    input_table.schema.column_names, "<RowID>"
                )
                pa_table = pa_table.rename_columns(
                    [rowID_name] + pa_table.column_names[1:]
                )

                pa_col = pa_table.column(self.input_col)
                df = pd.DataFrame.from_dict({"Chunks": pa_col.to_pandas()})

                df["Chunks"] = df["Chunks"].apply(
                    lambda row: splitter.split_text(row) if pd.notnull(row) else None
                )

                # Add row ID column to DataFrame to generate new row IDs
                df[rowID_name] = pa_table.column(rowID_name).to_pandas()

                # Introduce index for chunks within each row to generate new row IDs ("RowXX_YY")
                df["Internal Index"] = df["Chunks"].apply(
                    lambda row: list(range(1, len(row) + 1)) if row else [1]
                )

                ungrouped_df = df.explode(["Chunks", "Internal Index"])
                ungrouped_df["Chunks"] = ungrouped_df["Chunks"].astype("string")

                # Remove row ID column
                pa_table = pa_table.remove_column(0)

                # Create new output table with new row IDs ("RowXX_YY")
                ungrouped_df[rowID_name] = ungrouped_df[
                    [rowID_name, "Internal Index"]
                ].apply(
                    lambda row: f"""{row[rowID_name]}_{row["Internal Index"]}""",
                    axis=1,
                )
                row_ids = pa.Array.from_pandas(ungrouped_df[rowID_name])
                output_table = pa.Table.from_arrays([row_ids], names=[rowID_name])

                # Manually expand input table columns to match ungrouped column
                ungroup_indexes = self._create_ungroup_indexes(
                    ungrouped_df["Internal Index"]
                )

                for col in pa_table.column_names:
                    if (
                        col == self.input_col
                        and self.output_column == OutputColumnSetting.REPLACE.name
                    ):
                        output_table = output_table.append_column(
                            pa.field(self.input_col, pa.string()),
                            pa.Array.from_pandas(ungrouped_df["Chunks"]),
                        )
                    else:
                        take = pc.take(pa_table.column(col), ungroup_indexes)
                        output_table = output_table.append_column(col, take)

                if self.output_column == OutputColumnSetting.APPEND.name:
                    output_table = output_table.append_column(
                        append_column_field,
                        pa.Array.from_pandas(ungrouped_df["Chunks"]),
                    )

                yield knext.Table.from_pyarrow(output_table, row_ids="keep")

    def _create_ungroup_indexes(self, indexes: pd.Series):
        """Returns a list of indexes that specifies which elements to take from a column to ungroup it
        manually."""
        i = -1
        ungroup_indexes = []

        for x in indexes:
            if x == 1:
                i += 1
            ungroup_indexes.append(i)
        return ungroup_indexes
