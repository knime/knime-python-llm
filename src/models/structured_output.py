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

# Column name for tracking original row IDs when extracting multiple objects
ROW_ID_COLUMN = "Row ID"


class OutputFieldType(knext.EnumParameterOptions):
    """Types of fields that can be extracted from LLM responses."""
    
    String = (
        "String",
        "Text field.",
    )
    Integer = (
        "Integer",
        "Whole number field.",
    )
    Double = (
        "Double",
        "Floating point number field.",
    )
    Boolean = (
        "Boolean",
        "True/False field.",
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


@knext.parameter_group(label="Output field")
class OutputField:
    """Definition of a single field to extract from LLM responses."""
    
    name = knext.StringParameter(
        label="Field name",
        description="The name of the output field. This will be used as the column name in the output table.",
        default_value="",
    )

    field_type = knext.EnumParameter(
        label="Field type",
        description="The data type of this field. List types will create columns that can contain multiple values.",
        default_value=OutputFieldType.String.name,
        enum=OutputFieldType,
        style=knext.EnumParameter.Style.DROPDOWN,
    )

    description = knext.StringParameter(
        label="Description",
        description="A description of this field to help the model understand what to extract.",
        default_value="",
    )


@knext.parameter_group(label="Structured Output")
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
        label="Output fields",
        description="""Define the fields to extract from each prompt. Each field will be added as a separate column 
        in the output table. The model will be instructed to extract these fields from the input text.""",
        button_text="Add field",
        array_title="Output fields"
    )

    extract_multiple = knext.BoolParameter(
        label="Extract multiple objects",
        description="""If enabled, the model will extract multiple objects of the defined structure from each input row.
        Each extracted object will create a separate row in the output table, with all input columns duplicated.
        
        If no objects are extracted for an input row, one output row with missing values will be created.""",
        default_value=False,
    )


def validate_output_fields(output_fields):
    """
    Validate that output fields are properly configured.
    
    Args:
        output_fields: List of OutputField parameter groups
        
    Raises:
        knext.InvalidParametersError: If validation fails
    """
    if not output_fields:
        raise knext.InvalidParametersError(
            "At least one output field must be defined when using structured output format."
        )

    field_names = set()
    for i, field in enumerate(output_fields):
        if not field.name:
            raise knext.InvalidParametersError(
                f"Output field {i + 1} must have a name."
            )
        if not field.name.replace("_", "").replace(" ", "").isalnum():
            raise knext.InvalidParametersError(
                f"Output field name '{field.name}' must contain only letters, numbers, underscores, and spaces."
            )
        if field.name in field_names:
            raise knext.InvalidParametersError(
                f"Duplicate output field name: '{field.name}'. Each field must have a unique name."
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
    
    # If extract_multiple is enabled, wrap it in a list model
    if settings.extract_multiple:
        list_model_name = f"{model_name}List"
        return create_model(
            list_model_name,
            items=(TypingList[base_model], Field(description=f"List of {model_name} objects"))
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


def explode_lists(table, output_fields):
    """
    Explode list columns into multiple rows.
    
    Takes a table where output field columns contain lists and expands it so that:
    - Each list element gets its own row
    - Input columns are duplicated for each list element
    - All list columns must have the same length per row
    
    Args:
        table: PyArrow table with list columns to explode
        output_fields: List of OutputField parameter groups defining which columns contain lists
        
    Returns:
        PyArrow table with exploded rows
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    
    # Pick one of the list columns to derive the parent index mapping
    # Use the first output field column
    first_list_col_name = output_fields[0].name
    parent_indices = pc.list_parent_indices(table[first_list_col_name])
    
    # Flatten all list columns (the output field columns)
    list_column_names = {field.name for field in output_fields}
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
    
    # Build final exploded table (preserve original column order)
    return pa.table({**scalar_cols, **flattened_list_cols})

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
        responses: List of Pydantic model instances (or list wrapper models if extract_multiple=True)
        settings: StructuredOutputSettings parameter group
    
    Returns:
        PyArrow table where:
        - If extract_multiple=False: each row has scalar values
        - If extract_multiple=True: each row has list values (one list per input row containing extracted objects)
    """
    import pyarrow as pa

    if settings.extract_multiple:
        # When extract_multiple is enabled, each response is a wrapper model with an 'items' field
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
                # Create a list of values for each field
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
        # Single object per input row - original behavior
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


def add_structured_output_columns(input_schema, settings, add_row_id=False):
    """
    Add structured output columns to an input schema.
    
    Args:
        input_schema: Input table schema
        settings: StructuredOutputSettings parameter group
        add_row_id: Whether to add a row ID column (for extract_multiple)
        
    Returns:
        Schema with added structured output columns
    """
    import util
    
    output_schema = input_schema
    
    # Add row ID column when extract_multiple is enabled
    if add_row_id:
        row_id_col_name = util.handle_column_name_collision(
            output_schema.column_names, ROW_ID_COLUMN
        )
        output_schema = output_schema.append(
            knext.Column(ktype=knext.string(), name=row_id_col_name)
        )
    
    # Add field columns
    for field in settings.output_fields:
        column_name = util.handle_column_name_collision(
            output_schema.column_names, field.name
        )
        knime_type = get_output_field_knime_type(field.field_type)
        output_schema = output_schema.append(
            knext.Column(ktype=knime_type, name=column_name)
        )
    
    return output_schema
