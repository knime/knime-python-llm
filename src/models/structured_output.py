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


class OutputColumnType(knext.EnumParameterOptions):
    """Types of columns that can be extracted from LLM responses."""
    
    String = (
        "String",
        "Text column.",
    )
    Integer = (
        "Number (Integer)",
        "Integer column.",
    )
    Long = (
        "Number (Long Integer)",
        "Long integer column.",
    )
    Double = (
        "Number (Float)",
        "Floating point number column.",
    )
    Boolean = (
        "Boolean",
        "True/False column.",
    )


class OutputColumnQuantity(knext.EnumParameterOptions):
    """Quantity of values to extract for a single output column."""

    Single = (
        "Single",
        "Extract exactly one value of the defined type.",
    )
    Multiple = (
        "Multiple",
        "Extract multiple values (as a list) of the defined type.",
    )


@knext.parameter_group(label="Output column")
class OutputColumn:
    """Definition of a single column to extract from LLM responses."""
    
    name = knext.StringParameter(
        label="Column name",
        description="The name of the output column in the table.",
        default_value="",
    )

    column_type = knext.EnumParameter(
        label="Data type",
        description="The base data type of this column.",
        default_value=OutputColumnType.String.name,
        enum=OutputColumnType,
        style=knext.EnumParameter.Style.DROPDOWN,
    )

    quantity = knext.EnumParameter(
        label="Quantity",
        description="""Determines if this column should contain a single value 
        or multiple values (list) for each extracted item.""",
        default_value=OutputColumnQuantity.Single.name,
        enum=OutputColumnQuantity,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    description = knext.StringParameter(
        label="Description",
        description="A description of this column to help the model understand what to extract.",
        default_value="",
    )


class TargetObjectsPerInputRow(knext.EnumParameterOptions):
    """Number of target objects to extract per input row."""
    
    One = (
        "One",
        "Extract exactly one target object with the defined columns per input row.",
    )
    Multiple = (
        "Multiple",
        "Allow the model to extract one or more target objects with the defined structure per input row. "
        "The input columns will be duplicated for each extracted object.",
    )


@knext.parameter_group(label="Output Structure")
class StructuredOutputSettings:
    """Settings for structured output extraction."""
    
    target_object_name = knext.StringParameter(
        label="Target object name",
        description="""The name of the target object. This helps the model understand what it is extracting. 
        Invalid characters will be automatically converted to underscores.
        
        Example: 'Person Info', 'Product Details', 'Sentiment Analysis'""",
        default_value="ExtractedData",
    )

    target_object_description = knext.StringParameter(
        label="Target object description",
        description="""A description of what the target object represents. This provides context to the model 
        about the overall extraction task.
        
        Example: 'Information about a person including their name and age', 'Key details about a product'""",
        default_value="",
    )

    target_objects_per_input_row = knext.EnumParameter(
        label="Target objects per input row",
        description="Determines how many target objects are extracted for each input row.",
        default_value=TargetObjectsPerInputRow.One.name,
        enum=TargetObjectsPerInputRow,
        style=knext.EnumParameter.Style.VALUE_SWITCH,
    )

    input_row_id_column_name = knext.StringParameter(
        label="Input row ID column name",
        description="Name of the column that will store the original input row ID when multiple target objects are extracted per input row.",
        default_value="Input RowID",
    ).rule(
        knext.OneOf(target_objects_per_input_row, [TargetObjectsPerInputRow.Multiple.name]),
        knext.Effect.SHOW,
    )
    
    output_columns = knext.ParameterArray(
        parameters=OutputColumn(),
        label="Output columns",
        description="""Define the columns to extract from each prompt. Each column will be added to the output table. 
        The model will be instructed to extract these columns from the input text.""",
        button_text="Add column",
        array_title="Output columns"
    )



def get_resolved_column_names(input_column_names: list[str], settings: StructuredOutputSettings):
    """
    Get the resolved column names for structured output, handling collisions with input columns.
    
    Args:
        input_column_names: List of existing column names in the input table
        settings: StructuredOutputSettings parameter group
        
    Returns:
        Tuple of (resolved_input_row_id_name, resolved_output_column_names)
    """
    import util
    current_names = list(input_column_names)
    
    resolved_input_row_id_name = None
    if settings.target_objects_per_input_row == TargetObjectsPerInputRow.Multiple.name:
        resolved_input_row_id_name = util.handle_column_name_collision(
            current_names, settings.input_row_id_column_name
        )
        current_names.append(resolved_input_row_id_name)
        
    resolved_output_column_names = []
    for column in settings.output_columns:
        name = util.handle_column_name_collision(current_names, column.name)
        resolved_output_column_names.append(name)
        current_names.append(name)
        
    return resolved_input_row_id_name, resolved_output_column_names


def validate_output_columns(output_columns: list[OutputColumn]):
    """
    Validate that output columns are properly configured.
    
    Args:
        output_columns: List of OutputColumn parameter groups
        
    Raises:
        knext.InvalidParametersError: If validation fails
    """
    if not output_columns:
        raise knext.InvalidParametersError(
            "At least one output column must be defined when using structured output format."
        )

    column_names = set()
    for i, column in enumerate(output_columns):
        if not column.name:
            raise knext.InvalidParametersError(
                f"Output column {i + 1} must have a name."
            )
        if not column.name.replace("_", "").replace(" ", "").isalnum():
            raise knext.InvalidParametersError(
                f"Output column name '{column.name}' must contain only letters, numbers, underscores, and spaces."
            )
        if column.name in column_names:
            raise knext.InvalidParametersError(
                f"Duplicate output column name: '{column.name}'. Each column must have a unique name."
            )
        column_names.add(column.name)


def _sanitize_field_name(name: str) -> str:
    """Sanitize field name to match '^[a-zA-Z0-9_.-]{1,64}$'."""
    import re
    # Replace anything not a-z, A-Z, 0-9, _, . or - with _
    sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
    if not sanitized:
        sanitized = "field"
    return sanitized[:64]


def _get_llm_field_names(settings: StructuredOutputSettings):
    """Get unique sanitized field names for the LLM schema."""
    names = []
    seen = set()
    for column in settings.output_columns:
        base = _sanitize_field_name(column.name)
        candidate = base
        count = 1
        while candidate in seen:
            suffix = f"_{count}"
            candidate = base[:64 - len(suffix)] + suffix
            count += 1
        names.append(candidate)
        seen.add(candidate)
    return names


def create_pydantic_model(settings: StructuredOutputSettings):
    """
    Create a Pydantic model from the structured output settings.
    
    Args:
        settings: StructuredOutputSettings parameter group
        
    Returns:
        Pydantic model class for structured output
    """
    from pydantic import Field, create_model, BeforeValidator
    from typing import List as TypingList, Optional, Annotated

    def _coerce_to_int(v, min_val=None, max_val=None):
        """Coerce float or scientific notation string to integer and clip if necessary."""
        if v is None:
            return v
        res = v
        if isinstance(v, float):
            res = int(v)
        elif isinstance(v, str):
            try:
                # Handle scientific notation in strings
                res = int(float(v))
            except (ValueError, TypeError):
                pass
        
        if isinstance(res, int):
            if min_val is not None and res < min_val:
                return min_val
            if max_val is not None and res > max_val:
                return max_val
        return res

    Int32Type = Annotated[int, BeforeValidator(lambda v: _coerce_to_int(v, -2147483648, 2147483647))]
    Int64Type = Annotated[int, BeforeValidator(lambda v: _coerce_to_int(v, -9223372036854775808, 9223372036854775807))]

    # Map OutputColumnType to Python types
    type_mapping = {
        OutputColumnType.String.name: str,
        OutputColumnType.Integer.name: Int32Type,
        OutputColumnType.Long.name: Int64Type,
        OutputColumnType.Double.name: float,
        OutputColumnType.Boolean.name: bool,
    }

    # Build field definitions for Pydantic
    # All fields are optional to allow the LLM to omit values when information is missing
    field_definitions = {}
    llm_field_names = _get_llm_field_names(settings)
    for column, sanitized_name in zip(settings.output_columns, llm_field_names):
        python_type = type_mapping[column.column_type]
        if column.quantity == OutputColumnQuantity.Multiple.name:
            python_type = TypingList[python_type]
            
        field_description = column.description if column.description else column.name
        field_definitions[sanitized_name] = (
            Optional[python_type],
            Field(default=None, description=field_description),
        )

    # Use target_object_name as the model name (default: "ExtractedData")
    # Sanitize to comply with OpenAI's naming requirements (^[a-zA-Z0-9_-]+$)
    # Replace invalid characters with underscores
    import re
    model_name = settings.target_object_name if settings.target_object_name else "ExtractedData"
    model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
    
    # Use target_object_description as the model docstring if provided
    if settings.target_object_description:
        # The __doc__ attribute can be set via __config__ in Pydantic v2
        field_definitions["__doc__"] = settings.target_object_description

    # Create the Pydantic model dynamically
    base_model = create_model(model_name, **field_definitions)
    
    # If target_objects_per_input_row is Multiple, wrap it in a list model
    if settings.target_objects_per_input_row == TargetObjectsPerInputRow.Multiple.name:
        list_model_name = f"{model_name}List"
        return create_model(
            list_model_name,
            items=(TypingList[base_model], Field(description=f"List of {model_name} items"))
        )
    
    return base_model


def get_output_column_knime_type(column_type: str, quantity: str = OutputColumnQuantity.Single.name):
    """
    Map OutputColumnType to KNIME column type.
    
    Args:
        column_type: String name of OutputColumnType enum value
        quantity: String name of OutputColumnQuantity enum value
        
    Returns:
        KNIME column type
    """
    type_mapping = {
        OutputColumnType.String.name: knext.string(),
        OutputColumnType.Integer.name: knext.int32(),
        OutputColumnType.Long.name: knext.int64(),
        OutputColumnType.Double.name: knext.double(),
        OutputColumnType.Boolean.name: knext.bool_(),
    }
    
    knime_type = type_mapping[column_type]
    if quantity == OutputColumnQuantity.Multiple.name:
        return knext.list_(knime_type)
    return knime_type


def get_output_column_pyarrow_type(column_type: str, quantity: str = OutputColumnQuantity.Single.name):
    """
    Map OutputColumnType to PyArrow type.
    
    Args:
        column_type: String name of OutputColumnType enum value
        quantity: String name of OutputColumnQuantity enum value
        
    Returns:
        PyArrow type
    """
    import pyarrow as pa

    type_mapping = {
        OutputColumnType.String.name: pa.string(),
        OutputColumnType.Integer.name: pa.int32(),
        OutputColumnType.Long.name: pa.int64(),
        OutputColumnType.Double.name: pa.float64(),
        OutputColumnType.Boolean.name: pa.bool_(),
    }
    
    pa_type = type_mapping[column_type]
    if quantity == OutputColumnQuantity.Multiple.name:
        return pa.list_(pa_type)
    return pa_type


def _make_row_ids_unique(duplicated_row_ids, list_column):
    """
    Make RowIDs unique by appending an index for each exploded row.
    
    Args:
        duplicated_row_ids: PyArrow array of duplicated RowIDs after explosion
        list_column: PyArrow list column used to determine list lengths
        
    Returns:
        PyArrow array of unique RowIDs with appended indices (e.g., "Row0_0", "Row0_1")
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    
    # Calculate index within each group
    list_lengths = pc.list_value_length(list_column)
    
    # Create indices for each element within its list
    indices_within_group = []
    for length in list_lengths.to_pylist():
        indices_within_group.extend(range(length))
    
    # Append index to each RowID
    row_ids = duplicated_row_ids.to_pylist()
    unique_row_ids = [f"{row_id}_{idx}" for row_id, idx in zip(row_ids, indices_within_group)]
    return pa.array(unique_row_ids, type=pa.string())


def explode_lists(table, input_row_id_name: str, output_col_names: list[str]):
    """
    Explode list columns into multiple rows.
    
    Takes a table where output columns contain lists and expands it so that:
    - Each list element gets its own row
    - Input columns are duplicated for each list element
    - All list columns must have the same length per row
    
    Args:
        table: PyArrow table with list columns to explode
        input_row_id_name: Resolved name of the input row ID column
        output_col_names: List of resolved output column names
        
    Returns:
        PyArrow table with exploded rows
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    
    # Pick one of the list columns to derive the parent index mapping
    # Use the first output column
    first_list_col_name = output_col_names[0]
    first_list_col = table[first_list_col_name]
    parent_indices = pc.list_parent_indices(first_list_col)
    
    # Flatten all list columns (the output columns)
    list_column_names = set(output_col_names)
    flattened_list_cols = {
        name: pc.list_flatten(table[name])
        for name in table.column_names
        if name in list_column_names
    }
    
    # Duplicate scalar columns (input columns + RowID) using parent indices
    scalar_cols = {
        name: pc.take(table[name], parent_indices)
        for name in table.column_names
        if name not in list_column_names
    }
    
    # Make RowIDs unique by appending index (first column is assumed to be RowID)
    row_id_col_name = table.column_names[0]
    row_id_col = scalar_cols.pop(row_id_col_name)
    new_row_id_col = _make_row_ids_unique(row_id_col, first_list_col)
    
    # Build final exploded table (preserve original column order)
    return pa.table({row_id_col_name: new_row_id_col, **scalar_cols, input_row_id_name: row_id_col, **flattened_list_cols})

def create_empty(num_messages: int, output_column_names: list[str]):
    import pyarrow as pa
    empty_data = {
        name: [None] * num_messages for name in output_column_names
    }
    return pa.table(empty_data)


def structured_responses_to_table(responses, settings: StructuredOutputSettings, output_column_names: list[str]):
    """
    Convert a list of Pydantic model instances to a PyArrow table.
    
    Args:
        responses: List of Pydantic model instances (or list wrapper models if target_objects_per_input_row=Multiple)
        settings: StructuredOutputSettings parameter group
        output_column_names: List of resolved output column names
    
    Returns:
        PyArrow table where:
        - If target_objects_per_input_row=One: each row has scalar values
        - If target_objects_per_input_row=Multiple: each row has list values (one list per input row containing extracted items)
    """
    import pyarrow as pa
    llm_field_names = _get_llm_field_names(settings)

    if settings.target_objects_per_input_row == TargetObjectsPerInputRow.Multiple.name:
        # When Multiple is selected, each response is a wrapper model with an 'items' field
        # Create columns with lists (one list per input row)
        column_data = {sanitized_name: [] for sanitized_name in llm_field_names}
        
        for idx, response in enumerate(responses):
            # Get the list of items from the wrapper model
            items = getattr(response, 'items', [])
            
            # If no items extracted, add lists with single None value
            if not items:
                for sanitized_name in llm_field_names:
                    column_data[sanitized_name].append([None])
            else:
                # Create a list of values for each column
                for sanitized_name in llm_field_names:
                    column_values = [getattr(item, sanitized_name, None) for item in items]
                    column_data[sanitized_name].append(column_values)
        
        # Create PyArrow table with list columns
        arrays = []
        schema_fields = []
        
        for column, column_name, sanitized_name in zip(settings.output_columns, output_column_names, llm_field_names):
            inner_type = get_output_column_pyarrow_type(column.column_type, column.quantity)
            pa_type = pa.list_(inner_type)
            schema_fields.append(pa.field(column_name, pa_type))
            arrays.append(pa.array(column_data[sanitized_name], type=pa_type))
        
        schema = pa.schema(schema_fields)
        return pa.table(dict(zip(output_column_names, arrays)), schema=schema)
    else:
        # Single item per input row - original behavior
        column_data = {sanitized_name: [] for sanitized_name in llm_field_names}

        for response in responses:
            # response is a Pydantic model instance
            for sanitized_name in llm_field_names:
                value = getattr(response, sanitized_name, None)
                column_data[sanitized_name].append(value)

        # Create PyArrow table with appropriate types
        arrays = []
        schema_fields = []
        for column, column_name, sanitized_name in zip(settings.output_columns, output_column_names, llm_field_names):
            pa_type = get_output_column_pyarrow_type(column.column_type, column.quantity)
            schema_fields.append(pa.field(column_name, pa_type))
            arrays.append(pa.array(column_data[sanitized_name], type=pa_type))

        schema = pa.schema(schema_fields)
        return pa.table(dict(zip(output_column_names, arrays)), schema=schema)


def postprocess_table(input_table, result_table, settings: StructuredOutputSettings, input_row_id_name: str, output_col_names: list[str]):
    """
    Postprocess structured output by adding row IDs and optionally exploding list columns.
    
    Args:
        input_table: PyArrow table with input columns (including row IDs as first column)
        result_table: PyArrow table with structured output columns (from LLM)
        settings: StructuredOutputSettings parameter group
        input_row_id_name: Resolved name of the input row ID column
        output_col_names: List of resolved output column names
        
    Returns:
        PyArrow table with combined input and output columns, optionally exploded into multiple rows
    """
    import pyarrow as pa
    
    # Combine input and result columns
    combined_table = pa.Table.from_arrays(
        input_table.columns + result_table.columns,
        input_table.column_names + result_table.column_names,
    )
    
    # Handle row expansion for structured output with target_objects_per_input_row
    if settings.target_objects_per_input_row == TargetObjectsPerInputRow.Multiple.name:
        # Explode list columns into separate rows
        return explode_lists(combined_table, input_row_id_name, output_col_names)
    else:
        return combined_table


def add_structured_output_columns(input_schema: knext.Schema, settings: StructuredOutputSettings):
    """
    Add structured output columns to an input schema.
    
    Args:
        input_schema: Input table schema
        settings: StructuredOutputSettings parameter group
        
    Returns:
        Schema with added structured output columns
    """
    
    output_schema = input_schema
    
    input_row_id_name, output_col_names = get_resolved_column_names(
        input_schema.column_names, settings
    )
    
    # Add input row ID column when target_objects_per_input_row is Multiple
    if input_row_id_name:
        output_schema = output_schema.append(
            knext.Column(ktype=knext.string(), name=input_row_id_name)
        )
    
    # Add output columns
    for column, column_name in zip(settings.output_columns, output_col_names):
        knime_type = get_output_column_knime_type(column.column_type, column.quantity)
        output_schema = output_schema.append(
            knext.Column(ktype=knime_type, name=column_name)
        )
    
    return output_schema
