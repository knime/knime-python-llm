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
Tests for the structured_output module.
"""

import unittest
import pyarrow as pa
import knime.extension as knext
from pydantic import BaseModel, Field, create_model
from typing import List

from src.models import structured_output


class TestOutputFieldType(unittest.TestCase):
    """Test OutputFieldType enum values."""

    def test_string_type(self):
        self.assertEqual(structured_output.OutputFieldType.String.name, "String")
        
    def test_integer_type(self):
        self.assertEqual(structured_output.OutputFieldType.Integer.name, "Integer")
        
    def test_double_type(self):
        self.assertEqual(structured_output.OutputFieldType.Double.name, "Double")
        
    def test_boolean_type(self):
        self.assertEqual(structured_output.OutputFieldType.Boolean.name, "Boolean")
        
    def test_string_list_type(self):
        self.assertEqual(structured_output.OutputFieldType.StringList.name, "StringList")
        
    def test_integer_list_type(self):
        self.assertEqual(structured_output.OutputFieldType.IntegerList.name, "IntegerList")
        
    def test_double_list_type(self):
        self.assertEqual(structured_output.OutputFieldType.DoubleList.name, "DoubleList")
        
    def test_boolean_list_type(self):
        self.assertEqual(structured_output.OutputFieldType.BooleanList.name, "BooleanList")


class TestValidateOutputFields(unittest.TestCase):
    """Test validate_output_fields function."""

    def test_empty_fields_raises_error(self):
        with self.assertRaises(knext.InvalidParametersError) as cm:
            structured_output.validate_output_fields([])
        self.assertIn("At least one output column", str(cm.exception))

    def test_field_without_name_raises_error(self):
        field = structured_output.OutputField()
        field.name = ""
        field.field_type = structured_output.OutputFieldType.String.name
        
        with self.assertRaises(knext.InvalidParametersError) as cm:
            structured_output.validate_output_fields([field])
        self.assertIn("must have a name", str(cm.exception))

    def test_field_with_invalid_characters_raises_error(self):
        field = structured_output.OutputField()
        field.name = "invalid@name"
        field.field_type = structured_output.OutputFieldType.String.name
        
        with self.assertRaises(knext.InvalidParametersError) as cm:
            structured_output.validate_output_fields([field])
        self.assertIn("must contain only letters, numbers, underscores, and spaces", str(cm.exception))

    def test_duplicate_field_names_raises_error(self):
        field1 = structured_output.OutputField()
        field1.name = "same_name"
        field1.field_type = structured_output.OutputFieldType.String.name
        
        field2 = structured_output.OutputField()
        field2.name = "same_name"
        field2.field_type = structured_output.OutputFieldType.Integer.name
        
        with self.assertRaises(knext.InvalidParametersError) as cm:
            structured_output.validate_output_fields([field1, field2])
        self.assertIn("Duplicate output column name", str(cm.exception))

    def test_valid_fields_pass(self):
        field1 = structured_output.OutputField()
        field1.name = "field_one"
        field1.field_type = structured_output.OutputFieldType.String.name
        
        field2 = structured_output.OutputField()
        field2.name = "field two"
        field2.field_type = structured_output.OutputFieldType.Integer.name
        
        # Should not raise
        structured_output.validate_output_fields([field1, field2])


class TestCreatePydanticModel(unittest.TestCase):
    """Test create_pydantic_model function."""

    def test_create_simple_model_one_row(self):
        settings = structured_output.StructuredOutputSettings()
        settings.structure_name = "TestModel"
        settings.structure_description = "Test description"
        settings.output_rows_per_input_row = structured_output.OutputRowsPerInputRow.One.name
        
        field = structured_output.OutputField()
        field.name = "text_field"
        field.field_type = structured_output.OutputFieldType.String.name
        field.description = "A text field"
        
        settings.output_fields = [field]
        
        model = structured_output.create_pydantic_model(settings)
        
        # Check model name
        self.assertEqual(model.__name__, "TestModel")
        
        # Check fields exist
        self.assertIn("text_field", model.model_fields)

    def test_create_model_with_many_rows(self):
        settings = structured_output.StructuredOutputSettings()
        settings.structure_name = "ItemModel"
        settings.output_rows_per_input_row = structured_output.OutputRowsPerInputRow.Many.name
        
        field = structured_output.OutputField()
        field.name = "item_name"
        field.field_type = structured_output.OutputFieldType.String.name
        
        settings.output_fields = [field]
        
        model = structured_output.create_pydantic_model(settings)
        
        # Check model name has "List" suffix
        self.assertEqual(model.__name__, "ItemModelList")
        
        # Check it has an items field
        self.assertIn("items", model.model_fields)

    def test_create_model_with_multiple_field_types(self):
        settings = structured_output.StructuredOutputSettings()
        settings.structure_name = "MixedModel"
        settings.output_rows_per_input_row = structured_output.OutputRowsPerInputRow.One.name
        
        field1 = structured_output.OutputField()
        field1.name = "text"
        field1.field_type = structured_output.OutputFieldType.String.name
        
        field2 = structured_output.OutputField()
        field2.name = "number"
        field2.field_type = structured_output.OutputFieldType.Integer.name
        
        field3 = structured_output.OutputField()
        field3.name = "flag"
        field3.field_type = structured_output.OutputFieldType.Boolean.name
        
        settings.output_fields = [field1, field2, field3]
        
        model = structured_output.create_pydantic_model(settings)
        
        # Check all fields exist
        self.assertIn("text", model.model_fields)
        self.assertIn("number", model.model_fields)
        self.assertIn("flag", model.model_fields)


class TestGetOutputFieldKnimeType(unittest.TestCase):
    """Test get_output_field_knime_type function."""

    def test_string_type(self):
        result = structured_output.get_output_field_knime_type(
            structured_output.OutputFieldType.String.name
        )
        self.assertEqual(result, knext.string())

    def test_integer_type(self):
        result = structured_output.get_output_field_knime_type(
            structured_output.OutputFieldType.Integer.name
        )
        self.assertEqual(result, knext.int64())

    def test_double_type(self):
        result = structured_output.get_output_field_knime_type(
            structured_output.OutputFieldType.Double.name
        )
        self.assertEqual(result, knext.double())

    def test_boolean_type(self):
        result = structured_output.get_output_field_knime_type(
            structured_output.OutputFieldType.Boolean.name
        )
        self.assertEqual(result, knext.bool_())

    def test_string_list_type(self):
        result = structured_output.get_output_field_knime_type(
            structured_output.OutputFieldType.StringList.name
        )
        self.assertEqual(result, knext.list_(knext.string()))

    def test_integer_list_type(self):
        result = structured_output.get_output_field_knime_type(
            structured_output.OutputFieldType.IntegerList.name
        )
        self.assertEqual(result, knext.list_(knext.int64()))


class TestGetOutputFieldPyArrowType(unittest.TestCase):
    """Test get_output_field_pyarrow_type function."""

    def test_string_type(self):
        result = structured_output.get_output_field_pyarrow_type(
            structured_output.OutputFieldType.String.name
        )
        self.assertEqual(result, pa.string())

    def test_integer_type(self):
        result = structured_output.get_output_field_pyarrow_type(
            structured_output.OutputFieldType.Integer.name
        )
        self.assertEqual(result, pa.int64())

    def test_double_type(self):
        result = structured_output.get_output_field_pyarrow_type(
            structured_output.OutputFieldType.Double.name
        )
        self.assertEqual(result, pa.float64())

    def test_boolean_type(self):
        result = structured_output.get_output_field_pyarrow_type(
            structured_output.OutputFieldType.Boolean.name
        )
        self.assertEqual(result, pa.bool_())

    def test_string_list_type(self):
        result = structured_output.get_output_field_pyarrow_type(
            structured_output.OutputFieldType.StringList.name
        )
        self.assertEqual(result, pa.list_(pa.string()))


class TestMakeRowIdsUnique(unittest.TestCase):
    """Test _make_row_ids_unique function."""

    def test_simple_row_ids(self):
        # Create list column with different lengths
        list_column = pa.array([[1, 2], [3, 4, 5], [6]], type=pa.list_(pa.int64()))
        duplicated_row_ids = pa.array(["Row0", "Row0", "Row1", "Row1", "Row1", "Row2"])
        
        result = structured_output._make_row_ids_unique(duplicated_row_ids, list_column)
        
        expected = ["Row0_0", "Row0_1", "Row1_0", "Row1_1", "Row1_2", "Row2_0"]
        self.assertEqual(result.to_pylist(), expected)

    def test_single_element_lists(self):
        list_column = pa.array([[1], [2], [3]], type=pa.list_(pa.int64()))
        duplicated_row_ids = pa.array(["A", "B", "C"])
        
        result = structured_output._make_row_ids_unique(duplicated_row_ids, list_column)
        
        expected = ["A_0", "B_0", "C_0"]
        self.assertEqual(result.to_pylist(), expected)


class TestCreateEmpty(unittest.TestCase):
    """Test create_empty function."""

    def test_create_empty_table(self):
        settings = structured_output.StructuredOutputSettings()
        
        field1 = structured_output.OutputField()
        field1.name = "field1"
        field1.field_type = structured_output.OutputFieldType.String.name
        
        field2 = structured_output.OutputField()
        field2.name = "field2"
        field2.field_type = structured_output.OutputFieldType.Integer.name
        
        settings.output_fields = [field1, field2]
        
        result = structured_output.create_empty(settings, 3)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result.column_names, ["field1", "field2"])
        # All values should be None
        self.assertTrue(all(v is None for v in result["field1"].to_pylist()))
        self.assertTrue(all(v is None for v in result["field2"].to_pylist()))


class TestStructuredResponsesToTable(unittest.TestCase):
    """Test structured_responses_to_table function."""

    def test_convert_single_responses(self):
        settings = structured_output.StructuredOutputSettings()
        settings.output_rows_per_input_row = structured_output.OutputRowsPerInputRow.One.name
        
        field1 = structured_output.OutputField()
        field1.name = "name"
        field1.field_type = structured_output.OutputFieldType.String.name
        
        field2 = structured_output.OutputField()
        field2.name = "age"
        field2.field_type = structured_output.OutputFieldType.Integer.name
        
        settings.output_fields = [field1, field2]
        
        # Create mock Pydantic model instances
        model = structured_output.create_pydantic_model(settings)
        responses = [
            model(name="Alice", age=30),
            model(name="Bob", age=25),
        ]
        
        result = structured_output.structured_responses_to_table(responses, settings)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result["name"].to_pylist(), ["Alice", "Bob"])
        self.assertEqual(result["age"].to_pylist(), [30, 25])

    def test_convert_many_responses(self):
        settings = structured_output.StructuredOutputSettings()
        settings.structure_name = "Item"
        settings.output_rows_per_input_row = structured_output.OutputRowsPerInputRow.Many.name
        
        field1 = structured_output.OutputField()
        field1.name = "item"
        field1.field_type = structured_output.OutputFieldType.String.name
        
        settings.output_fields = [field1]
        
        # In Many mode, create_pydantic_model returns a wrapper model with an 'items' field
        # that contains a list of the actual item models
        wrapper_model = structured_output.create_pydantic_model(settings)
        
        # Create the base item model that matches the output fields
        # This simulates what the LLM would return
        ItemModel = create_model("Item", item=(str, Field(description="item")))
        
        # Create responses where each response has a list of items
        responses = [
            wrapper_model(items=[ItemModel(item="apple"), ItemModel(item="banana")]),
            wrapper_model(items=[ItemModel(item="carrot")]),
        ]
        
        result = structured_output.structured_responses_to_table(responses, settings)
        
        self.assertEqual(len(result), 2)
        # Each row should have a list
        self.assertEqual(result["item"].to_pylist(), [["apple", "banana"], ["carrot"]])


class TestExplodeLists(unittest.TestCase):
    """Test explode_lists function."""

    def test_explode_simple_lists(self):
        settings = structured_output.StructuredOutputSettings()
        settings.input_row_id_column_name = "Input Row ID"
        
        field1 = structured_output.OutputField()
        field1.name = "items"
        field1.field_type = structured_output.OutputFieldType.StringList.name
        
        settings.output_fields = [field1]
        
        # Create input table with list column and Row ID
        table = pa.table({
            "Row ID": ["0", "1"],
            "input_col": ["a", "b"],
            "items": [["x", "y"], ["z"]],
        })
        
        result = structured_output.explode_lists(table, settings)
        
        # Should have 3 rows (2 from first, 1 from second)
        self.assertEqual(len(result), 3)
        # Check Row IDs are unique
        row_ids = result["Row ID"].to_pylist()
        self.assertEqual(row_ids, ["0_0", "0_1", "1_0"])
        # Check input column is duplicated
        self.assertEqual(result["input_col"].to_pylist(), ["a", "a", "b"])
        # Check items are flattened
        self.assertEqual(result["items"].to_pylist(), ["x", "y", "z"])
        # Check Input Row ID column exists
        self.assertIn("Input Row ID", result.column_names)


class TestPostprocessTable(unittest.TestCase):
    """Test postprocess_table function."""

    def test_postprocess_one_row_per_input(self):
        settings = structured_output.StructuredOutputSettings()
        settings.output_rows_per_input_row = structured_output.OutputRowsPerInputRow.One.name
        
        field = structured_output.OutputField()
        field.name = "output"
        field.field_type = structured_output.OutputFieldType.String.name
        settings.output_fields = [field]
        
        input_table = pa.table({
            "input_col": [1, 2],
        })
        
        result_table = pa.table({
            "output": ["a", "b"],
        })
        
        result = structured_output.postprocess_table(input_table, result_table, settings)
        
        self.assertEqual(len(result), 2)
        self.assertIn("input_col", result.column_names)
        self.assertIn("output", result.column_names)

    def test_postprocess_many_rows_per_input(self):
        settings = structured_output.StructuredOutputSettings()
        settings.output_rows_per_input_row = structured_output.OutputRowsPerInputRow.Many.name
        settings.input_row_id_column_name = "Input Row ID"
        
        field = structured_output.OutputField()
        field.name = "output"
        field.field_type = structured_output.OutputFieldType.StringList.name
        settings.output_fields = [field]
        
        input_table = pa.table({
            "input_col": [1, 2],
        })
        
        result_table = pa.table({
            "output": [["a", "b"], ["c"]],
        })
        
        result = structured_output.postprocess_table(input_table, result_table, settings)
        
        # Should be exploded to 3 rows
        self.assertEqual(len(result), 3)
        self.assertIn("Input Row ID", result.column_names)


class TestAddStructuredOutputColumns(unittest.TestCase):
    """Test add_structured_output_columns function."""

    def test_add_columns_one_row(self):
        settings = structured_output.StructuredOutputSettings()
        settings.output_rows_per_input_row = structured_output.OutputRowsPerInputRow.One.name
        
        field = structured_output.OutputField()
        field.name = "new_col"
        field.field_type = structured_output.OutputFieldType.String.name
        settings.output_fields = [field]
        
        input_schema = knext.Schema.from_columns([
            knext.Column(knext.int64(), "existing_col")
        ])
        
        result = structured_output.add_structured_output_columns(input_schema, settings)
        
        self.assertEqual(len(result), 2)
        self.assertIn("new_col", result.column_names)

    def test_add_columns_many_rows(self):
        settings = structured_output.StructuredOutputSettings()
        settings.output_rows_per_input_row = structured_output.OutputRowsPerInputRow.Many.name
        settings.input_row_id_column_name = "Input Row ID"
        
        field = structured_output.OutputField()
        field.name = "new_col"
        field.field_type = structured_output.OutputFieldType.String.name
        settings.output_fields = [field]
        
        input_schema = knext.Schema.from_columns([
            knext.Column(knext.int64(), "existing_col")
        ])
        
        result = structured_output.add_structured_output_columns(input_schema, settings)
        
        # Should add Input Row ID column and new_col
        self.assertEqual(len(result), 3)
        self.assertIn("Input Row ID", result.column_names)
        self.assertIn("new_col", result.column_names)


if __name__ == "__main__":
    unittest.main()
