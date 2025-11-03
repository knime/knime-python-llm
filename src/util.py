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
from typing import Callable, List, Union, Tuple
from abc import ABC, abstractmethod
import re
import io
import pyarrow as pa
import pyarrow.compute as pc
from dataclasses import dataclass


def is_nominal(column: knext.Column) -> bool:
    # Filter nominal columns
    return column.ktype == knext.string() or column.ktype == knext.bool_()


def create_type_filter(*ktypes: knext.KnimeType) -> Callable[[knext.Column], bool]:
    return lambda c: c.ktype in ktypes


ai_icon = "icons/generic/brain.png"

main_category = knext.category(
    path="/labs",
    level_id="kai",
    name="AI",
    description="",
    icon=ai_icon,
)


def check_canceled(ctx: knext.ExecutionContext) -> None:
    if ctx.is_canceled():
        raise RuntimeError("Execution canceled.")


def pick_default_column(input_table: knext.Schema, ktype: knext.KnimeType) -> str:
    default_column = pick_default_columns(input_table, ktype, 1)[0]
    return default_column


def pick_default_columns(
    input_table: knext.Schema, ktype: knext.KnimeType, n_columns: int
) -> List[str]:
    columns = [c for c in input_table if c.ktype == ktype]

    if len(columns) < n_columns:
        raise knext.InvalidParametersError(
            f"The input table does not contain enough ({n_columns}) distinct columns of type '{str(ktype)}'. Found: {len(columns)}"
        )

    return [column_name.name for column_name in columns[n_columns * -1 :]]


def check_column(
    input_table: knext.Schema,
    column_name: str,
    expected_types: Union[knext.KnimeType, List[knext.KnimeType]],
    column_purpose: str,
    table_name: str = "input table",
) -> None:
    """
    Raises an InvalidParametersError if a column named column_name is not contained in input_table or has the wrong KnimeType.
    """
    if column_name not in input_table.column_names:
        raise knext.InvalidParametersError(
            f"The {column_purpose} column '{column_name}' is missing in the {table_name}."
        )
    ktype = input_table[column_name].ktype

    if not isinstance(expected_types, list):
        expected_types = [expected_types]

    if ktype not in expected_types:
        raise knext.InvalidParametersError(
            f"The {column_purpose} column '{column_name}' is of type {str(ktype)} but should be one of the types {str(expected_types)}."
        )


def handle_column_name_collision(
    column_names: List[str],
    column_name: str,
) -> str:
    """
    If the output column name collides with an input column name, it's been made unique by appending (#<count>).
    For example, if "column" exists as an input column name and is entered as an output column name,
    the output column name will be "column (#1)". Adding "column" or "column (#1)" again will result
    in "column1 (#2)".
    """
    basename = column_name.strip()
    existing_column_names = set(column_names)

    if basename not in existing_column_names:
        return basename

    # Pattern to match strings that have a name followed by a
    # numerical identifier in parentheses, e.g. "column (#1)"
    pattern = re.compile(r"^(.*) \(#(\d+)\)$")
    match = pattern.match(basename)
    if match:
        basename, index = match.groups()
        index = int(index) + 1
    else:
        index = 1

    new_column_name = basename
    while new_column_name in existing_column_names:
        new_column_name = f"{basename} (#{index})"
        index += 1

    return new_column_name


class MissingValueHandlingOptions(knext.EnumParameterOptions):
    SkipRow = (
        "Skip rows",
        "Rows with missing values will be ignored.",
    )
    Fail = (
        "Fail",
        "This node will fail during the execution.",
    )


class MissingValueOutputOptions(knext.EnumParameterOptions):
    """
    Instead of skipping rows with missing values as done with MissingValueHandlingOptions,
    this class outputs missing values. This ensures table consistency by maintaining
    the original row count, preventing the confusion caused by sudden table shrinkage
    for nodes that perform row-wise mapping operations (e.g. Text Embedder).
    """

    OutputMissingValues = (
        "Output missing values",
        "Rows with missing values will not be processed but are included in the output.",
    )
    Fail = (
        "Fail",
        "This node will fail during the execution.",
    )


def skip_missing_values(df, col_name: str, ctx: knext.ExecutionContext):
    import pandas as pd

    df: pd.DataFrame = df
    # Drops rows with missing values
    df_cleaned = df.dropna(subset=[col_name], how="any")
    n_skipped_rows = len(df) - len(df_cleaned)

    if n_skipped_rows > 0:
        ctx.set_warning(f"{n_skipped_rows} / {len(df)} rows are skipped.")

    return df_cleaned


def handle_missing_and_empty_values(
    df,
    input_column: str,
    missing_value_handling_setting: MissingValueHandlingOptions,
    ctx: knext.ExecutionContext,
    check_empty_values: bool = True,
):
    import pandas as pd

    df: pd.DataFrame = df
    # Drops rows if SkipRow option is selected, otherwise fails
    # if there are any missing values in the input column (=Fail option is selected)
    has_missing_values = df[input_column].isna().any()
    if (
        missing_value_handling_setting == MissingValueHandlingOptions.SkipRow
        and has_missing_values
    ):
        df = skip_missing_values(df, input_column, ctx)
    elif has_missing_values:
        missing_row_id = df[df[input_column].isnull()].index[0]
        raise ValueError(
            f"There are missing values in column {input_column}. See row ID <{missing_row_id}> for the first row that contains a missing value."
        )

    if df.empty:
        raise ValueError("All rows are skipped due to missing values.")

    if check_empty_values:
        # Check for empty values
        for id, value in df[input_column].items():
            if not value.strip():
                raise ValueError(
                    f"Empty values are not supported. See row ID {id} for the first empty value."
                )

    return df


def message_type() -> knext.LogicalType:
    from knime.types.message import MessageValue

    return knext.logical(MessageValue)


def to_human_message(message):
    from knime.types.message import MessageValue, to_langchain_message
    import langchain_core.messages as lcm
    import json

    if isinstance(message, MessageValue):
        return to_langchain_message(message)
    elif isinstance(message, (dict, list)):
        return lcm.HumanMessage(json.dumps(message))
    return lcm.HumanMessage(message)


class BaseMapper(ABC):
    def __init__(
        self,
        columns: Union[str, List[str]],
        fn: Callable[[pa.Table], pa.Table],
    ) -> None:
        super().__init__()
        self._columns = [columns] if isinstance(columns, str) else columns
        self._fn = fn

    def _is_valid(self, pa_array: pa.Array) -> pa.Array:
        is_null_array = pc.is_null(pa_array)
        is_not_null_array = pc.invert(is_null_array)

        # Check if the array's type is a string type
        if pa.types.is_string(pa_array.type) or pa.types.is_large_string(pa_array.type):
            # For string types, trim whitespace and check for non-empty
            text_array_trimmed = pc.utf8_trim_whitespace(pa_array)
            is_empty_string = pc.equal(pc.utf8_length(text_array_trimmed), 0)
            # A string is valid if it's not null AND not an empty string
            return pc.fill_null(
                pc.and_(is_not_null_array, pc.invert(is_empty_string)), False
            )

        else:
            # For any other data type, just check if it's not null.
            return is_not_null_array

    @abstractmethod
    def map(self, table: pa.Table) -> pa.Array:
        """Maps the given table to an output array."""

    @property
    @abstractmethod
    def all_missing(self) -> bool:
        """Indicates if all observed values were missing or empty"""


class FailOnMissingMapper(BaseMapper):
    def map(self, table: pa.Table):
        for column in self._columns:
            text_array = table.column(column)
            is_valid = self._is_valid(text_array)
            # min_count = 0 allows for empty tables
            all_valid = pc.all(is_valid, min_count=0)
            if not all_valid.as_py():
                raise ValueError(
                    f"There are missing or empty values in column {column}. "
                    f"See row ID <{self._get_row_id_of_first_null(table, is_valid)}> "
                    "for the first row that contains such a value."
                )

        return self._fn(table)

    def _get_row_id_of_first_null(self, table, is_valid):
        empties = pc.filter(table, pc.invert(is_valid))
        return empties[0][0].as_py()

    @property
    def all_missing(self) -> bool:
        return False


class OutputMissingMapper(BaseMapper):
    def __init__(
        self,
        columns: Union[str, List[str]],
        fn: Callable,
    ) -> None:
        super().__init__(columns, fn)
        self._all_missing = True

    @property
    def all_missing(self):
        return self._all_missing

    def map(self, table: pa.Table):
        is_valid = self._compute_validity(table)
        if pc.any(is_valid).as_py() or table.num_rows == 0:
            self._all_missing = False

        all_valid = pc.all(is_valid).as_py()
        valid_table = table if all_valid else table.filter(is_valid)

        result_table: pa.Table = self._fn(valid_table)

        if all_valid:
            return result_table

        return self._reconstruct_result_table(result_table, is_valid)

    def _compute_validity(self, table: pa.Table) -> pa.Array:
        """
        Computes the validity of the specified columns (self._columns) in the table.

        The result is a boolean array where each element is True if all specified column values
        in that row are not missing values, and False if any of the specified columns in that row
        are missing.
        """

        validity_arrays = [
            self._is_valid(table.column(column)) for column in self._columns
        ]

        is_valid = validity_arrays[0]
        for arr in validity_arrays[1:]:
            is_valid = pc.and_(is_valid, arr)
        is_valid = is_valid.combine_chunks()

        return is_valid

    def _reconstruct_result_table(
        self, result_table: pa.Table, column_validity: pa.Array
    ) -> pa.Table:
        """
        Reconstructs the result table while maintaining the original row structure.

        This method ensures that the output table has the same number of rows as the input table,
        filling in missing values where necessary based on the column validity mask.

        `replace_with_mask` currently does not support nested lists (ListArrays), making it unsuitable for our use case
        where embeddings column contain lists of floats. As an alternative, we use `take` to select valid values.
        """

        valid_indices = self._compute_valid_indices(column_validity)

        expanded_arrays = []
        for array in result_table:
            valid_values = pc.take(array, valid_indices)
            expanded_arrays.append(valid_values)

        return pa.table(expanded_arrays, schema=result_table.schema)

    def _compute_valid_indices(self, column_validity: pa.Array) -> pa.Array:
        # Convert boolean validity mask to integers (1 for valid rows, 0 for invalid).
        validity_int = pc.cast(column_validity, pa.int64())

        # Compute cumulative sum to determine valid row positions.
        cumulative_indices = pc.cumulative_sum(validity_int)

        # Adjust indices by subtracting 1 to get the correct row positions.
        valid_indices = pc.subtract(cumulative_indices, 1)

        return pc.if_else(column_validity, valid_indices, None)


def table_column_adapter(
    table: pa.Table,
    fn: Callable[[List[str]], List],
    input_column: str,
    output_column: pa.field,
) -> pa.Table:
    """
    Adapter function that takes a pyarrow table and returns a pyarrow table
    containing only the processed output column.
    """
    if input_column not in table.schema.names:
        raise ValueError(f"Column '{input_column}' not found in the input table.")

    input_values = table.column(input_column).to_pylist()
    processed_values = fn(input_values)

    output_array = pa.array(processed_values)
    output_schema = pa.schema([(output_column.name, output_column.type)])

    return pa.table([output_array], schema=output_schema)


async def abatched_apply(afn, inputs: list, batch_size: int) -> list:
    outputs = []
    for batch in _generate_batches(inputs, batch_size):
        outputs.extend(await afn(batch))
    return outputs


def batched_apply(fn: Callable[[list], list], inputs: list, batch_size: int) -> list:
    outputs = []
    for batch in _generate_batches(inputs, batch_size):
        outputs.extend(fn(batch))
    return outputs


def _generate_batches(inputs: list, batch_size: int):
    for i in range(0, len(inputs), batch_size):
        yield inputs[i : i + batch_size]


class ProgressTracker:
    def __init__(self, total_rows: int, ctx: knext.ExecutionContext):
        self.total_rows = total_rows
        self.current_progress = 0
        self.ctx = ctx

    def update_progress(self, batch_size: int):
        check_canceled(self.ctx)
        self.current_progress += batch_size
        self.ctx.set_progress(self.current_progress / self.total_rows)


@dataclass
class OutputColumn:
    """Stores information on an output column of a node."""

    default_name: str
    knime_type: knext.KnimeType
    pa_type: pa.DataType

    def to_knime_column(self):
        column = knext.Column(self.knime_type, self.default_name)
        return column


def create_empty_table(
    table: knext.Table, output_columns: List[OutputColumn]
) -> knext.Table:
    """Constructs an empty KNIME Table with the correct output columns."""
    if table is None:
        pa_table = pa.table([])
    else:
        pa_table = table.to_pyarrow()

    for col in output_columns:
        output_column_name = handle_column_name_collision(
            table.schema.column_names if table is not None else [], col.default_name
        )
        pa_table = pa_table.append_column(
            output_column_name,
            pa.array([], col.pa_type),
        )
    return knext.Table.from_pyarrow(pa_table)


def image_table_present(ctx: knext.DialogCreationContext) -> bool:
    """Check if an image table is connected."""
    specs = ctx.get_input_specs()
    return len(specs) == 2 and specs[1] is not None


def image_column_filter(column: knext.Column) -> bool:
    from PIL import Image

    img_type = knext.logical(Image.Image)
    return column.ktype == img_type


def prepare_images(
    image_table: knext.Table,
    image_columns: knext.ColumnFilterConfig,
) -> List[Tuple[str, bytes, str]]:
    """
    Extracts images from table according to selected image columns,
    converts them to PNG in-memory file tuples, and returns a list
    of (filename, raw_png_bytes, mimetype).
    This format is required by the OpenAI Image API for image generation.
    """
    image_column_names = _get_image_column_names(
        image_columns,
        schema=image_table.schema,
    )

    image_df = image_table[image_column_names].to_pandas()

    files = []
    for idx, row in enumerate(image_df.itertuples(index=False), start=1):
        # for each image column, save the image to a bytes buffer
        for i, col in enumerate(image_column_names):
            # create an in-memory bytes buffer to hold the PNG data
            buffer = io.BytesIO()

            # pull out the PIL.Image object from the row
            pil_img = row[i]

            # save the image into our buffer in PNG format
            # this encodes the PIL.Image into valid PNG bytes
            pil_img.save(buffer, format="PNG")

            # extract raw PNG bytes from the buffer
            raw = buffer.getvalue()

            filename = f"{col}_{idx}.png"
            files.append((filename, raw, "image/png"))
    return files


def _get_image_column_names(
    column_filter_config,
    schema: knext.Schema,
) -> List[str]:
    """
    Get the image column names from the column filter config and schema.
    """
    if not column_filter_config:
        return []
    return [col.name for col in column_filter_config.apply(schema)]
