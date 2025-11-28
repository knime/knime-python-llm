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

"""
Structured output extraction for LLM nodes.

This module provides functionality for extracting structured data from LLM responses,
including parameter definitions, Pydantic model creation, and table conversion utilities.
"""

import knime.extension as knext

# Column name for tracking original row IDs when creating multiple output rows
ROW_ID_COLUMN = "Row ID"


class OutputFieldType(knext.EnumParameterOptions):
    """Types of columns that can be extracted from LLM responses."""
    
    String = (
        "String",
        "Text column.",
    )
    Integer = (
        "Integer",
        "Whole number column.",
    )
    Double = (
        "Double",
        "Floating point number column.",
    )
    Boolean = (
        "Boolean",
        "True/False column.",
    )
    StringList = (
        "String List",
        "List of text values.",
    )
    IntegerList = (
        "Integer List",
        "List of whole numbers.",
    )
    DoubleList = (
        "Double List",
        "List of floating point numbers.",
    )
    BooleanList = (
        "Boolean List",
        "List of True/False values.",
    )


@knext.parameter_group(label="Output column")
class OutputField:
    """Definition of a single column to extract from LLM responses."""
    
    name = knext.StringParameter(
        label="Column name",
        description="The name of the output column in the table.",
        default_value="",
    )

    field_type = knext.EnumParameter(
        label="Data type",
        description="The data type of this column. List types can contain multiple values per row.",
        default_value=OutputFieldType.String.name,
        enum=OutputFieldType,
        style=knext.EnumParameter.Style.DROPDOWN,
    )

    description = knext.StringParameter(
        label="Description",
        description="A description of this column to help the model understand what to extract.",
        default_value="",
    )


class OutputRowsPerInputRow(knext.EnumParameterOptions):
    """Number of output rows to create per input row."""
    
    One = (
        "One",
        "Extract exactly one output row with the defined columns per input row.",
    )
    Many = (
        "Many",
        "Allow the model to extract one or more output rows with the defined structure per input row. "
        "The input columns will be duplicated for each extracted row.",
    )


@knext.parameter_group(label="Output Structure")
class StructuredOutputSettings:
    """Settings for structured output extraction."""
    
    structure_name = knext.StringParameter(
        label="Structure name",
        description="""The name of the output structure. This helps the model understand what it is extracting.
        
        Example: 'PersonInfo', 'ProductDetails', 'SentimentAnalysis'""",
        default_value="ExtractedData",
    )

    structure_description = knext.StringParameter(
        label="Structure description",
        description="""A description of what the output structure represents. This provides context to the model 
        about the overall extraction task.
        
        Example: 'Information about a person including their name and age', 'Key details about a product'""",
        default_value="",
    )

    output_fields = knext.ParameterArray(
        parameters=OutputField(),
        label="Output columns",
        description="""Define the columns to extract from each prompt. Each column will be added to the output table. 
        The model will be instructed to extract these columns from the input text.""",
        button_text="Add column",
        array_title="Output columns"
    )

    output_rows_per_input_row = knext.EnumParameter(
        label="Output rows per input row",
        description="Determines how many output rows are created for each input row.",
        default_value=OutputRowsPerInputRow.One.name,
        enum=OutputRowsPerInputRow,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    input_row_id_column_name = knext.StringParameter(
        label="Input row ID column name",
        description="Name of the column that will store the original input row ID when multiple output rows are created per input row.",
        default_value="Input Row ID",
    ).rule(
        knext.OneOf(output_rows_per_input_row, [OutputRowsPerInputRow.Many.name]),
        knext.Effect.SHOW,
    )


def validate_output_fields(output_fields):
    """
    Validate that output columns are properly configured.
    
    Args:
        output_fields: List of OutputField parameter groups
        
    Raises:
        knext.InvalidParametersError: If validation fails
    """
    if not output_fields:
        raise knext.InvalidParametersError(
            "At least one output column must be defined when using structured output format."
        )

    field_names = set()
    for i, field in enumerate(output_fields):
        if not field.name:
            raise knext.InvalidParametersError(
                f"Output column {i + 1} must have a name."
            )
        if not field.name.replace("_", "").replace(" ", "").isalnum():
            raise knext.InvalidParametersError(
                f"Output column name '{field.name}' must contain only letters, numbers, underscores, and spaces."
            )
        if field.name in field_names:
            raise knext.InvalidParametersError(
                f"Duplicate output column name: '{field.name}'. Each column must have a unique name."
            )
        field_names.add(field.name)


def create_pydantic_model(settings):
    """
    Create a Pydantic model from the structured output settings.
    
    Args:
        settings: StructuredOutputSettings parameter group
        
    Returns:
        Pydantic model class for structured output
    """
    from pydantic import Field, create_model
    from typing import List as TypingList

    # Map OutputFieldType to Python types
    type_mapping = {
        OutputFieldType.String.name: str,
        OutputFieldType.Integer.name: int,
        OutputFieldType.Double.name: float,
        OutputFieldType.Boolean.name: bool,
        OutputFieldType.StringList.name: TypingList[str],
        OutputFieldType.IntegerList.name: TypingList[int],
        OutputFieldType.DoubleList.name: TypingList[float],
        OutputFieldType.BooleanList.name: TypingList[bool],
    }

    # Build field definitions for Pydantic
    field_definitions = {}
    for field in settings.output_fields:
        python_type = type_mapping[field.field_type]
        field_description = field.description if field.description else field.name
        field_definitions[field.name] = (
            python_type,
            Field(description=field_description),
        )

    # Use structure_name as the model name (default: "ExtractedData")
    model_name = settings.structure_name if settings.structure_name else "ExtractedData"
    
    # Use structure_description as the model docstring if provided
    if settings.structure_description:
        # The __doc__ attribute can be set via __config__ in Pydantic v2
        field_definitions["__doc__"] = settings.structure_description

    # Create the Pydantic model dynamically
    base_model = create_model(model_name, **field_definitions)
    
    # If output_rows_per_input_row is Many, wrap it in a list model
    if settings.output_rows_per_input_row == OutputRowsPerInputRow.Many.name:
        list_model_name = f"{model_name}List"
        return create_model(
            list_model_name,
            items=(TypingList[base_model], Field(description=f"List of {model_name} items"))
        )
    
    return base_model


def get_output_field_knime_type(field_type: str):
    """
    Map OutputFieldType to KNIME column type.
    
    Args:
        field_type: String name of OutputFieldType enum value
        
    Returns:
        KNIME column type
    """
    type_mapping = {
        OutputFieldType.String.name: knext.string(),
        OutputFieldType.Integer.name: knext.int64(),
        OutputFieldType.Double.name: knext.double(),
        OutputFieldType.Boolean.name: knext.bool_(),
        OutputFieldType.StringList.name: knext.list_(knext.string()),
        OutputFieldType.IntegerList.name: knext.list_(knext.int64()),
        OutputFieldType.DoubleList.name: knext.list_(knext.double()),
        OutputFieldType.BooleanList.name: knext.list_(knext.bool_()),
    }
    return type_mapping[field_type]


def get_output_field_pyarrow_type(field_type: str):
    """
    Map OutputFieldType to PyArrow type.
    
    Args:
        field_type: String name of OutputFieldType enum value
        
    Returns:
        PyArrow type
    """
    import pyarrow as pa

    type_mapping = {
        OutputFieldType.String.name: pa.string(),
        OutputFieldType.Integer.name: pa.int64(),
        OutputFieldType.Double.name: pa.float64(),
        OutputFieldType.Boolean.name: pa.bool_(),
        OutputFieldType.StringList.name: pa.list_(pa.string()),
        OutputFieldType.IntegerList.name: pa.list_(pa.int64()),
        OutputFieldType.DoubleList.name: pa.list_(pa.float64()),
        OutputFieldType.BooleanList.name: pa.list_(pa.bool_()),
    }
    return type_mapping[field_type]


def _make_row_ids_unique(duplicated_row_ids, list_column):
    """
    Make Row IDs unique by appending an index for each exploded row.
    
    Args:
        duplicated_row_ids: PyArrow array of duplicated Row IDs after explosion
        list_column: PyArrow list column used to determine list lengths
        
    Returns:
        PyArrow array of unique Row IDs with appended indices (e.g., "Row0_0", "Row0_1")
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    
    # Calculate index within each group
    list_lengths = pc.list_value_length(list_column)
    
    # Create indices for each element within its list
    indices_within_group = []
    for length in list_lengths.to_pylist():
        indices_within_group.extend(range(length))
    
    # Append index to each Row ID
    row_ids = duplicated_row_ids.to_pylist()
    unique_row_ids = [f"{row_id}_{idx}" for row_id, idx in zip(row_ids, indices_within_group)]
    return pa.array(unique_row_ids, type=pa.string())


def explode_lists(table, settings: StructuredOutputSettings):
    """
    Explode list columns into multiple rows.
    
    Takes a table where output columns contain lists and expands it so that:
    - Each list element gets its own row
    - Input columns are duplicated for each list element
    - All list columns must have the same length per row
    
    Args:
        table: PyArrow table with list columns to explode
        settings: StructuredOutputSettings parameter group
        
    Returns:
        PyArrow table with exploded rows
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import util
    
    # Get the input row ID column name from settings
    input_row_id_column_name = util.handle_column_name_collision(
        table.column_names, settings.input_row_id_column_name
    )
    
    # Pick one of the list columns to derive the parent index mapping
    # Use the first output column
    first_list_col_name = settings.output_fields[0].name
    first_list_col = table[first_list_col_name]
    parent_indices = pc.list_parent_indices(first_list_col)
    
    # Flatten all list columns (the output columns)
    list_column_names = {field.name for field in settings.output_fields}
    flattened_list_cols = {
        name: pc.list_flatten(table[name])
        for name in table.column_names
        if name in list_column_names
    }
    
    # Duplicate scalar columns (input columns + Row ID) using parent indices
    scalar_cols = {
        name: pc.take(table[name], parent_indices)
        for name in table.column_names
        if name not in list_column_names
    }
    
    # Make Row IDs unique by appending index (first column is assumed to be Row ID)
    row_id_col_name = table.column_names[0]
    row_id_col = scalar_cols.pop(row_id_col_name)
    new_row_id_col = _make_row_ids_unique(row_id_col, first_list_col)
    
    # Build final exploded table (preserve original column order)
    return pa.table({row_id_col_name: new_row_id_col, **scalar_cols, input_row_id_column_name: row_id_col, **flattened_list_cols})

def create_empty(settings: StructuredOutputSettings, num_messages: int):
    import pyarrow as pa
    empty_data = {
        field.name: [None] * num_messages for field in settings.output_fields
    }
    return pa.table(empty_data)


def structured_responses_to_table(responses, settings):
    """
    Convert a list of Pydantic model instances to a PyArrow table.
    
    Args:
        responses: List of Pydantic model instances (or list wrapper models if output_rows_per_input_row=Many)
        settings: StructuredOutputSettings parameter group
    
    Returns:
        PyArrow table where:
        - If output_rows_per_input_row=One: each row has scalar values
        - If output_rows_per_input_row=Many: each row has list values (one list per input row containing extracted items)
    """
    import pyarrow as pa

    if settings.output_rows_per_input_row == OutputRowsPerInputRow.Many.name:
        # When Many is selected, each response is a wrapper model with an 'items' field
        # Create columns with lists (one list per input row)
        column_data = {field.name: [] for field in settings.output_fields}
        
        for idx, response in enumerate(responses):
            # Get the list of items from the wrapper model
            items = getattr(response, 'items', [])
            
            # If no items extracted, add lists with single None value
            if not items:
                for field in settings.output_fields:
                    column_data[field.name].append([None])
            else:
                # Create a list of values for each column
                for field in settings.output_fields:
                    field_values = [getattr(item, field.name, None) for item in items]
                    column_data[field.name].append(field_values)
        
        # Create PyArrow table with list columns
        arrays = []
        schema_fields = []
        
        for field in settings.output_fields:
            inner_type = get_output_field_pyarrow_type(field.field_type)
            schema_fields.append(pa.field(field.name, pa.list_(inner_type)))
            arrays.append(pa.array(column_data[field.name]))
        
        schema = pa.schema(schema_fields)
        column_names = [f.name for f in schema_fields]
        return pa.table(dict(zip(column_names, arrays)), schema=schema)
    else:
        # Single item per input row - original behavior
        column_data = {field.name: [] for field in settings.output_fields}

        for response in responses:
            # response is a Pydantic model instance
            for field in settings.output_fields:
                value = getattr(response, field.name, None)
                column_data[field.name].append(value)

        # Create PyArrow table with appropriate types
        arrays = []
        schema_fields = []
        for field in settings.output_fields:
            pa_type = get_output_field_pyarrow_type(field.field_type)
            schema_fields.append(pa.field(field.name, pa_type))
            arrays.append(pa.array(column_data[field.name], type=pa_type))

        schema = pa.schema(schema_fields)
        return pa.table(dict(zip([f.name for f in settings.output_fields], arrays)), schema=schema)


def postprocess_table(input_table, result_table, settings):
    """
    Postprocess structured output by adding row IDs and optionally exploding list columns.
    
    Args:
        input_table: PyArrow table with input columns
        result_table: PyArrow table with structured output columns (from LLM)
        settings: StructuredOutputSettings parameter group
        
    Returns:
        PyArrow table with combined input and output columns, optionally exploded into multiple rows
    """
    import pyarrow as pa
    import util
    
    # Add row ID column for structured output with output_rows_per_input_row
    if settings.output_rows_per_input_row == OutputRowsPerInputRow.Many.name:
        row_id_col_name = util.handle_column_name_collision(
            input_table.column_names, ROW_ID_COLUMN
        )
        # Create row IDs as strings (batch row indices)
        row_ids = pa.array([str(i) for i in range(len(input_table))], type=pa.string())
        input_table = input_table.append_column(row_id_col_name, row_ids)
    
    # Combine input and result columns
    combined_table = pa.Table.from_arrays(
        input_table.columns + result_table.columns,
        input_table.column_names + result_table.column_names,
    )
    
    # Handle row expansion for structured output with output_rows_per_input_row
    if settings.output_rows_per_input_row == OutputRowsPerInputRow.Many.name:
        # Explode list columns into separate rows
        return explode_lists(combined_table, settings)
    else:
        return combined_table


def add_structured_output_columns(input_schema, settings):
    """
    Add structured output columns to an input schema.
    
    Args:
        input_schema: Input table schema
        settings: StructuredOutputSettings parameter group
        
    Returns:
        Schema with added structured output columns
    """
    import util
    
    output_schema = input_schema
    
    # Add input row ID column when output_rows_per_input_row is Many
    # This column stores the original input row ID before explosion
    if settings.output_rows_per_input_row == OutputRowsPerInputRow.Many.name:
        input_row_id_col_name = util.handle_column_name_collision(
            output_schema.column_names, settings.input_row_id_column_name
        )
        output_schema = output_schema.append(
            knext.Column(ktype=knext.string(), name=input_row_id_col_name)
        )
    
    # Add output columns
    for field in settings.output_fields:
        column_name = util.handle_column_name_collision(
            output_schema.column_names, field.name
        )
        knime_type = get_output_field_knime_type(field.field_type)
        output_schema = output_schema.append(
            knext.Column(ktype=knime_type, name=column_name)
        )
    
    return output_schema
