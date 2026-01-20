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
from models import structured_output


# Mock classes to work around ParameterGroupHolder issues when setting parameter arrays directly
class MockOutputColumn:
    """Mock for OutputColumn parameter group."""
    def __init__(self, name="", column_type="String", quantity="Single", description=""):
        self.name = name
        self.column_type = column_type
        self.quantity = quantity
        self.description = description


class MockStructuredOutputSettings:
    """Mock for StructuredOutputSettings parameter group."""
    def __init__(self):
        self.target_object_name = "Target object"
        self.target_object_description = ""
        self.output_columns = []
        self.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.One.name
        self.input_row_id_column_name = "Input Row ID"


class TestOutputColumnType(unittest.TestCase):
    """Test OutputColumnType enum values."""

    def test_string_type(self):
        self.assertEqual(structured_output.OutputColumnType.String.name, "String")
        
    def test_integer_type(self):
        self.assertEqual(structured_output.OutputColumnType.Integer.name, "Integer")

    def test_long_type(self):
        self.assertEqual(structured_output.OutputColumnType.Long.name, "Long")
        
    def test_double_type(self):
        self.assertEqual(structured_output.OutputColumnType.Double.name, "Double")
        
    def test_boolean_type(self):
        self.assertEqual(structured_output.OutputColumnType.Boolean.name, "Boolean")


class TestOutputColumnQuantity(unittest.TestCase):
    """Test OutputColumnQuantity enum values."""

    def test_single_quantity(self):
        self.assertEqual(structured_output.OutputColumnQuantity.Single.name, "Single")

    def test_multiple_quantity(self):
        self.assertEqual(structured_output.OutputColumnQuantity.Multiple.name, "Multiple")


class TestValidateOutputColumns(unittest.TestCase):
    """Test validate_output_columns function."""

    def test_empty_columns_raises_error(self):
        with self.assertRaises(knext.InvalidParametersError) as cm:
            structured_output.validate_output_columns([])
        self.assertIn("At least one output column", str(cm.exception))

    def test_column_without_name_raises_error(self):
        column = MockOutputColumn()
        column.name = ""
        column.column_type = structured_output.OutputColumnType.String.name
        
        with self.assertRaises(knext.InvalidParametersError) as cm:
            structured_output.validate_output_columns([column])
        self.assertIn("must have a name", str(cm.exception))

    def test_column_with_invalid_characters_raises_error(self):
        column = MockOutputColumn()
        column.name = "invalid@name"
        column.column_type = structured_output.OutputColumnType.String.name
        
        with self.assertRaises(knext.InvalidParametersError) as cm:
            structured_output.validate_output_columns([column])
        self.assertIn("must contain only letters, numbers, underscores, and spaces", str(cm.exception))

    def test_duplicate_column_names_raises_error(self):
        column1 = MockOutputColumn()
        column1.name = "same_name"
        column1.column_type = structured_output.OutputColumnType.String.name
        
        column2 = MockOutputColumn()
        column2.name = "same_name"
        column2.column_type = structured_output.OutputColumnType.Integer.name
        
        with self.assertRaises(knext.InvalidParametersError) as cm:
            structured_output.validate_output_columns([column1, column2])
        self.assertIn("Duplicate output column name", str(cm.exception))

    def test_valid_columns_pass(self):
        column1 = MockOutputColumn()
        column1.name = "column_one"
        column1.column_type = structured_output.OutputColumnType.String.name
        
        column2 = MockOutputColumn()
        column2.name = "column two"
        column2.column_type = structured_output.OutputColumnType.Integer.name
        
        # Should not raise
        structured_output.validate_output_columns([column1, column2])


class TestCreatePydanticModel(unittest.TestCase):
    """Test create_pydantic_model function."""

    def test_create_simple_model_one_row(self):
        settings = MockStructuredOutputSettings()
        settings.target_object_name = "TestModel"
        settings.target_object_description = "Test description"
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.One.name
        
        field = MockOutputColumn()
        field.name = "text_field"
        field.column_type = structured_output.OutputColumnType.String.name
        field.description = "A text field"
        
        settings.output_columns = [field]
        
        model = structured_output.create_pydantic_model(settings)
        
        # Check model name
        self.assertEqual(model.__name__, "TestModel")
        
        # Check fields exist
        self.assertIn("text_field", model.model_fields)

    def test_create_model_with_multiple_rows(self):
        settings = MockStructuredOutputSettings()
        settings.target_object_name = "ItemModel"
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.Multiple.name
        
        field = MockOutputColumn()
        field.name = "item_name"
        field.column_type = structured_output.OutputColumnType.String.name
        
        settings.output_columns = [field]
        
        model = structured_output.create_pydantic_model(settings)
        
        # Check model name has "List" suffix
        self.assertEqual(model.__name__, "ItemModelList")
        
        # Check it has an items field
        self.assertIn("items", model.model_fields)

    def test_int_coercion(self):
        """Test that large numbers in scientific notation (float or string) are coerced to int and clipped to limits."""
        settings = MockStructuredOutputSettings()
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.One.name
        
        field = MockOutputColumn()
        field.name = "large_id"
        field.column_type = structured_output.OutputColumnType.Long.name
        
        settings.output_columns = [field]
        
        model = structured_output.create_pydantic_model(settings)
        
        # Test float input within range
        val_in_range = 1.0e15
        m1 = model(large_id=val_in_range)
        self.assertEqual(m1.large_id, int(val_in_range))
        
        # Test float input out of range (greater than max int64)
        # 9.223372036854776e+18 is > 2^63 - 1
        large_val = 9.223372036854776e+18
        m2 = model(large_id=large_val)
        self.assertEqual(m2.large_id, 9223372036854775807)
        
        # Test string input with scientific notation out of range
        m3 = model(large_id="1e30")
        self.assertEqual(m3.large_id, 9223372036854775807)

        # Test small out of range
        m4 = model(large_id="-1e30")
        self.assertEqual(m4.large_id, -9223372036854775808)

        # Test Integer (32-bit) clipping
        field.column_type = structured_output.OutputColumnType.Integer.name
        model_int32 = structured_output.create_pydantic_model(settings)
        m5 = model_int32(large_id=1e12)
        self.assertEqual(m5.large_id, 2147483647)

    def test_create_model_with_multiple_field_types(self):
        settings = MockStructuredOutputSettings()
        settings.target_object_name = "MixedModel"
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.One.name
        
        field1 = MockOutputColumn()
        field1.name = "text"
        field1.column_type = structured_output.OutputColumnType.String.name
        
        field2 = MockOutputColumn()
        field2.name = "number"
        field2.column_type = structured_output.OutputColumnType.Integer.name
        
        field3 = MockOutputColumn()
        field3.name = "flag"
        field3.column_type = structured_output.OutputColumnType.Boolean.name
        
        field4 = MockOutputColumn()
        field4.name = "ratio"
        field4.column_type = structured_output.OutputColumnType.Double.name

        field5 = MockOutputColumn()
        field5.name = "big_id"
        field5.column_type = structured_output.OutputColumnType.Long.name

        settings.output_columns = [field1, field2, field3, field4, field5]
        
        model = structured_output.create_pydantic_model(settings)
        
        # Check all fields exist
        self.assertIn("text", model.model_fields)
        self.assertIn("number", model.model_fields)
        self.assertIn("flag", model.model_fields)
        self.assertIn("ratio", model.model_fields)
        self.assertIn("big_id", model.model_fields)

    def test_target_object_name_with_spaces(self):
        """Test that spaces in target_object_name are converted to underscores."""
        settings = MockStructuredOutputSettings()
        settings.target_object_name = "Person Info"
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.One.name
        
        field = MockOutputColumn()
        field.name = "name"
        field.column_type = structured_output.OutputColumnType.String.name
        
        settings.output_columns = [field]
        
        model = structured_output.create_pydantic_model(settings)
        
        # Check that model name has underscores instead of spaces
        self.assertEqual(model.__name__, "Person_Info")

    def test_target_object_name_with_multiple_spaces(self):
        """Test that multiple spaces are all converted to underscores."""
        settings = MockStructuredOutputSettings()
        settings.target_object_name = "Product Detail Summary"
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.One.name
        
        field = MockOutputColumn()
        field.name = "product"
        field.column_type = structured_output.OutputColumnType.String.name
        
        settings.output_columns = [field]
        
        model = structured_output.create_pydantic_model(settings)
        
        # Check that all spaces are converted to underscores
        self.assertEqual(model.__name__, "Product_Detail_Summary")

    def test_target_object_name_with_special_characters(self):
        """Test that special characters are converted to underscores."""
        settings = MockStructuredOutputSettings()
        settings.target_object_name = "Person@Info#123"
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.One.name
        
        field = MockOutputColumn()
        field.name = "name"
        field.column_type = structured_output.OutputColumnType.String.name
        
        settings.output_columns = [field]
        
        model = structured_output.create_pydantic_model(settings)
        
        # Check that special characters are converted to underscores
        self.assertEqual(model.__name__, "Person_Info_123")

    def test_target_object_name_with_allowed_characters(self):
        """Test that allowed characters (letters, numbers, underscores, hyphens) are preserved."""
        settings = MockStructuredOutputSettings()
        settings.target_object_name = "Person-Info_123"
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.One.name
        
        field = MockOutputColumn()
        field.name = "name"
        field.column_type = structured_output.OutputColumnType.String.name
        
        settings.output_columns = [field]
        
        model = structured_output.create_pydantic_model(settings)
        
        # Check that allowed characters are preserved
        self.assertEqual(model.__name__, "Person-Info_123")


class TestFieldSanitization(unittest.TestCase):
    """Test field name sanitization for LLM compatibility."""

    def test_sanitize_field_name(self):
        self.assertEqual(structured_output._sanitize_field_name("Full ID"), "Full_ID")
        self.assertEqual(structured_output._sanitize_field_name("Invalid@Char#"), "Invalid_Char_")
        self.assertEqual(structured_output._sanitize_field_name("dots.and-dashes_work"), "dots.and-dashes_work")
        self.assertEqual(structured_output._sanitize_field_name(""), "field")
        # Test length limit
        long_name = "a" * 100
        self.assertEqual(len(structured_output._sanitize_field_name(long_name)), 64)

    def test_get_llm_field_names_uniqueness(self):
        settings = MockStructuredOutputSettings()
        col1 = MockOutputColumn(name="Full ID")
        col2 = MockOutputColumn(name="Full!ID")
        col3 = MockOutputColumn(name="Other")
        settings.output_columns = [col1, col2, col3]

        names = structured_output._get_llm_field_names(settings)
        self.assertEqual(names, ["Full_ID", "Full_ID_1", "Other"])
        self.assertEqual(len(set(names)), 3)


class TestGetOutputColumnKnimeType(unittest.TestCase):
    """Test get_output_column_knime_type function."""

    def test_string_type(self):
        result = structured_output.get_output_column_knime_type(
            structured_output.OutputColumnType.String.name
        )
        self.assertEqual(result, knext.string())

    def test_integer_type(self):
        result = structured_output.get_output_column_knime_type(
            structured_output.OutputColumnType.Integer.name
        )
        self.assertEqual(result, knext.int32())

    def test_long_type(self):
        result = structured_output.get_output_column_knime_type(
            structured_output.OutputColumnType.Long.name
        )
        self.assertEqual(result, knext.int64())

    def test_double_type(self):
        result = structured_output.get_output_column_knime_type(
            structured_output.OutputColumnType.Double.name
        )
        self.assertEqual(result, knext.double())

    def test_boolean_type(self):
        result = structured_output.get_output_column_knime_type(
            structured_output.OutputColumnType.Boolean.name
        )
        self.assertEqual(result, knext.bool_())

    def test_string_list_type(self):
        result = structured_output.get_output_column_knime_type(
            structured_output.OutputColumnType.String.name,
            structured_output.OutputColumnQuantity.Multiple.name
        )
        self.assertEqual(result, knext.list_(knext.string()))

    def test_integer_list_type(self):
        result = structured_output.get_output_column_knime_type(
            structured_output.OutputColumnType.Integer.name,
            structured_output.OutputColumnQuantity.Multiple.name
        )
        self.assertEqual(result, knext.list_(knext.int32()))

    def test_long_list_type(self):
        result = structured_output.get_output_column_knime_type(
            structured_output.OutputColumnType.Long.name,
            structured_output.OutputColumnQuantity.Multiple.name
        )
        self.assertEqual(result, knext.list_(knext.int64()))

    def test_double_list_type(self):
        result = structured_output.get_output_column_knime_type(
            structured_output.OutputColumnType.Double.name,
            structured_output.OutputColumnQuantity.Multiple.name
        )
        self.assertEqual(result, knext.list_(knext.double()))

    def test_boolean_list_type(self):
        result = structured_output.get_output_column_knime_type(
            structured_output.OutputColumnType.Boolean.name,
            structured_output.OutputColumnQuantity.Multiple.name
        )
        self.assertEqual(result, knext.list_(knext.bool_()))


class TestGetOutputColumnPyArrowType(unittest.TestCase):
    """Test get_output_column_pyarrow_type function."""

    def test_string_type(self):
        result = structured_output.get_output_column_pyarrow_type(
            structured_output.OutputColumnType.String.name
        )
        self.assertEqual(result, pa.string())

    def test_integer_type(self):
        result = structured_output.get_output_column_pyarrow_type(
            structured_output.OutputColumnType.Integer.name
        )
        self.assertEqual(result, pa.int32())

    def test_long_type(self):
        result = structured_output.get_output_column_pyarrow_type(
            structured_output.OutputColumnType.Long.name
        )
        self.assertEqual(result, pa.int64())

    def test_double_type(self):
        result = structured_output.get_output_column_pyarrow_type(
            structured_output.OutputColumnType.Double.name
        )
        self.assertEqual(result, pa.float64())

    def test_boolean_type(self):
        result = structured_output.get_output_column_pyarrow_type(
            structured_output.OutputColumnType.Boolean.name
        )
        self.assertEqual(result, pa.bool_())

    def test_string_list_type(self):
        result = structured_output.get_output_column_pyarrow_type(
            structured_output.OutputColumnType.String.name,
            structured_output.OutputColumnQuantity.Multiple.name
        )
        self.assertEqual(result, pa.list_(pa.string()))

    def test_integer_list_type(self):
        result = structured_output.get_output_column_pyarrow_type(
            structured_output.OutputColumnType.Integer.name,
            structured_output.OutputColumnQuantity.Multiple.name
        )
        self.assertEqual(result, pa.list_(pa.int32()))

    def test_long_list_type(self):
        result = structured_output.get_output_column_pyarrow_type(
            structured_output.OutputColumnType.Long.name,
            structured_output.OutputColumnQuantity.Multiple.name
        )
        self.assertEqual(result, pa.list_(pa.int64()))

    def test_double_list_type(self):
        result = structured_output.get_output_column_pyarrow_type(
            structured_output.OutputColumnType.Double.name,
            structured_output.OutputColumnQuantity.Multiple.name
        )
        self.assertEqual(result, pa.list_(pa.float64()))

    def test_boolean_list_type(self):
        result = structured_output.get_output_column_pyarrow_type(
            structured_output.OutputColumnType.Boolean.name,
            structured_output.OutputColumnQuantity.Multiple.name
        )
        self.assertEqual(result, pa.list_(pa.bool_()))


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
        settings = MockStructuredOutputSettings()
        
        field1 = MockOutputColumn()
        field1.name = "field1"
        field1.column_type = structured_output.OutputColumnType.String.name
        
        field2 = MockOutputColumn()
        field2.name = "field2"
        field2.column_type = structured_output.OutputColumnType.Integer.name
        
        settings.output_columns = [field1, field2]
        
        result = structured_output.create_empty(3, ["field1", "field2"])
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result.column_names, ["field1", "field2"])
        # All values should be None
        self.assertTrue(all(v is None for v in result["field1"].to_pylist()))
        self.assertTrue(all(v is None for v in result["field2"].to_pylist()))


class TestStructuredResponsesToTable(unittest.TestCase):
    """Test structured_responses_to_table function."""

    def test_convert_single_responses(self):
        settings = MockStructuredOutputSettings()
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.One.name
        
        field1 = MockOutputColumn()
        field1.name = "name"
        field1.column_type = structured_output.OutputColumnType.String.name
        
        field2 = MockOutputColumn()
        field2.name = "age"
        field2.column_type = structured_output.OutputColumnType.Integer.name
        
        settings.output_columns = [field1, field2]
        
        # Create mock Pydantic model instances
        model = structured_output.create_pydantic_model(settings)
        responses = [
            model(name="Alice", age=30),
            model(name="Bob", age=25),
        ]
        
        result = structured_output.structured_responses_to_table(responses, settings, ["name", "age"])
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result["name"].to_pylist(), ["Alice", "Bob"])
        self.assertEqual(result["age"].to_pylist(), [30, 25])

    def test_convert_single_responses_all_types(self):
        """Test conversion of single responses with all column types."""
        settings = MockStructuredOutputSettings()
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.One.name
        
        columns = [
            MockOutputColumn("str", structured_output.OutputColumnType.String.name),
            MockOutputColumn("int", structured_output.OutputColumnType.Integer.name),
            MockOutputColumn("long", structured_output.OutputColumnType.Long.name),
            MockOutputColumn("double", structured_output.OutputColumnType.Double.name),
            MockOutputColumn("bool", structured_output.OutputColumnType.Boolean.name),
        ]
        settings.output_columns = columns
        resolved_names = [c.name for c in columns]
        
        model = structured_output.create_pydantic_model(settings)
        responses = [
            model(**{"str": "A", "int": 1, "long": 10**12, "double": 1.5, "bool": True}),
            model(**{"str": "B", "int": 2, "long": 20**12, "double": 2.5, "bool": False}),
        ]
        
        result = structured_output.structured_responses_to_table(responses, settings, resolved_names)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result["str"].to_pylist(), ["A", "B"])
        self.assertEqual(result["int"].to_pylist(), [1, 2])
        self.assertEqual(result["long"].to_pylist(), [10**12, 20**12])
        self.assertEqual(result["double"].to_pylist(), [1.5, 2.5])
        self.assertEqual(result["bool"].to_pylist(), [True, False])

    def test_convert_multiple_responses(self):
        settings = MockStructuredOutputSettings()
        settings.target_object_name = "Item"
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.Multiple.name
        
        field1 = MockOutputColumn()
        field1.name = "item"
        field1.column_type = structured_output.OutputColumnType.String.name
        
        settings.output_columns = [field1]
        
        # In Multiple mode, create_pydantic_model returns a wrapper model with an 'items' field
        # that contains a list of the actual item models
        wrapper_model = structured_output.create_pydantic_model(settings)
        
        # Extract the item type from the wrapper model to ensure we use the correct model
        items_field = wrapper_model.model_fields['items']
        ItemModel = items_field.annotation.__args__[0]  # Get the List[ItemModel] inner type
        
        # Create responses where each response has a list of items
        responses = [
            wrapper_model(items=[ItemModel(item="apple"), ItemModel(item="banana")]),
            wrapper_model(items=[ItemModel(item="carrot")]),
        ]
        
        result = structured_output.structured_responses_to_table(responses, settings, ["item"])
        
        self.assertEqual(len(result), 2)
        # Each row should have a list
        self.assertEqual(result["item"].to_pylist(), [["apple", "banana"], ["carrot"]])

    def test_convert_multiple_responses_all_types(self):
        """Test conversion of multiple responses per row with all column types."""
        settings = MockStructuredOutputSettings()
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.Multiple.name
        
        columns = [
            MockOutputColumn("str", structured_output.OutputColumnType.String.name),
            MockOutputColumn("int", structured_output.OutputColumnType.Integer.name),
            MockOutputColumn("long", structured_output.OutputColumnType.Long.name),
            MockOutputColumn("double", structured_output.OutputColumnType.Double.name),
            MockOutputColumn("bool", structured_output.OutputColumnType.Boolean.name),
        ]
        settings.output_columns = columns
        resolved_names = [c.name for c in columns]
        
        wrapper_model = structured_output.create_pydantic_model(settings)
        items_field = wrapper_model.model_fields['items']
        ItemModel = items_field.annotation.__args__[0]
        
        responses = [
            wrapper_model(items=[
                ItemModel(**{"str": "A1", "int": 1, "long": 10**12, "double": 1.1, "bool": True}),
                ItemModel(**{"str": "A2", "int": 2, "long": 20**12, "double": 1.2, "bool": False}),
            ]),
            wrapper_model(items=[
                ItemModel(**{"str": "B1", "int": 3, "long": 30**12, "double": 2.1, "bool": True}),
            ]),
        ]
        
        result = structured_output.structured_responses_to_table(responses, settings, resolved_names)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result["str"].to_pylist(), [["A1", "A2"], ["B1"]])
        self.assertEqual(result["int"].to_pylist(), [[1, 2], [3]])
        self.assertEqual(result["long"].to_pylist(), [[10**12, 20**12], [30**12]])
        self.assertEqual(result["double"].to_pylist(), [[1.1, 1.2], [2.1]])
        self.assertEqual(result["bool"].to_pylist(), [[True, False], [True]])

    def test_missing_values_in_responses(self):
        """Test that missing values are properly handled when information cannot be extracted."""
        settings = MockStructuredOutputSettings()
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.One.name
        
        field1 = MockOutputColumn()
        field1.name = "name"
        field1.column_type = structured_output.OutputColumnType.String.name
        
        field2 = MockOutputColumn()
        field2.name = "age"
        field2.column_type = structured_output.OutputColumnType.Integer.name
        
        field3 = MockOutputColumn()
        field3.name = "city"
        field3.column_type = structured_output.OutputColumnType.String.name
        
        settings.output_columns = [field1, field2, field3]
        
        # Create mock Pydantic model instances with some missing values
        model = structured_output.create_pydantic_model(settings)
        responses = [
            model(name="Alice", age=30, city="NYC"),  # All fields present
            model(name="Bob", age=None, city="LA"),   # Age missing
            model(name=None, age=25, city=None),      # Name and city missing
        ]
        
        result = structured_output.structured_responses_to_table(responses, settings, ["name", "age", "city"])
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result["name"].to_pylist(), ["Alice", "Bob", None])
        self.assertEqual(result["age"].to_pylist(), [30, None, 25])
        self.assertEqual(result["city"].to_pylist(), ["NYC", "LA", None])

    def test_missing_values_in_multiple_mode(self):
        """Test that missing values work correctly in Multiple mode."""
        settings = MockStructuredOutputSettings()
        settings.target_object_name = "Person"
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.Multiple.name
        
        field1 = MockOutputColumn()
        field1.name = "name"
        field1.column_type = structured_output.OutputColumnType.String.name
        
        field2 = MockOutputColumn()
        field2.name = "age"
        field2.column_type = structured_output.OutputColumnType.Integer.name
        
        settings.output_columns = [field1, field2]
        
        wrapper_model = structured_output.create_pydantic_model(settings)
        items_field = wrapper_model.model_fields['items']
        PersonModel = items_field.annotation.__args__[0]
        
        # Create responses with some missing values
        responses = [
            wrapper_model(items=[
                PersonModel(name="Alice", age=30),
                PersonModel(name="Bob", age=None),  # Age missing
            ]),
            wrapper_model(items=[
                PersonModel(name=None, age=25),  # Name missing
            ]),
        ]
        
        result = structured_output.structured_responses_to_table(responses, settings, ["name", "age"])
        
        self.assertEqual(len(result), 2)
        # Check that None values are preserved in lists
        self.assertEqual(result["name"].to_pylist(), [["Alice", "Bob"], [None]])
        self.assertEqual(result["age"].to_pylist(), [[30, None], [25]])


class TestExplodeLists(unittest.TestCase):
    """Test explode_lists function."""

    def test_explode_simple_lists(self):
        settings = MockStructuredOutputSettings()
        settings.input_row_id_column_name = "Input Row ID"
        
        field1 = MockOutputColumn()
        field1.name = "items"
        field1.column_type = structured_output.OutputColumnType.String.name
        field1.quantity = structured_output.OutputColumnQuantity.Multiple.name
        
        settings.output_columns = [field1]
        
        # Create input table with list column and Row ID
        table = pa.table({
            "Row ID": ["0", "1"],
            "input_col": ["a", "b"],
            "items": [["x", "y"], ["z"]],
        })
        
        result = structured_output.explode_lists(table, "Input Row ID", ["items"])
        
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
        settings = MockStructuredOutputSettings()
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.One.name
        
        field = MockOutputColumn()
        field.name = "output"
        field.column_type = structured_output.OutputColumnType.String.name
        settings.output_columns = [field]
        
        input_table = pa.table({
            "input_col": [1, 2],
        })
        
        result_table = pa.table({
            "output": ["a", "b"],
        })
        
        result = structured_output.postprocess_table(input_table, result_table, settings, None, ["output"])
        
        self.assertEqual(len(result), 2)
        self.assertIn("input_col", result.column_names)
        self.assertIn("output", result.column_names)

    def test_postprocess_multiple_rows_per_input(self):
        settings = MockStructuredOutputSettings()
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.Multiple.name
        settings.input_row_id_column_name = "Input Row ID"
        
        field = MockOutputColumn()
        field.name = "output"
        field.column_type = structured_output.OutputColumnType.String.name
        field.quantity = structured_output.OutputColumnQuantity.Multiple.name
        settings.output_columns = [field]
        
        # First column must be Row ID as per explode_lists logic
        input_table = pa.table({
            "Row ID": ["0", "1"],
            "input_col": [1, 2],
        })
        
        result_table = pa.table({
            "output": [["a", "b"], ["c"]],
        })
        
        result = structured_output.postprocess_table(input_table, result_table, settings, "Input Row ID", ["output"])
        
        # Should be exploded to 3 rows
        self.assertEqual(len(result), 3)
        self.assertIn("Input Row ID", result.column_names)


class TestAddStructuredOutputColumns(unittest.TestCase):
    """Test add_structured_output_columns function."""

    def test_add_columns_one_row(self):
        settings = MockStructuredOutputSettings()
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.One.name
        
        field = MockOutputColumn()
        field.name = "new_col"
        field.column_type = structured_output.OutputColumnType.String.name
        settings.output_columns = [field]

        
        input_schema = knext.Schema.from_columns([
            knext.Column(knext.int64(), "existing_col")
        ])
        
        result = structured_output.add_structured_output_columns(input_schema, settings)
        
        self.assertEqual(len(list(result)), 2)
        self.assertIn("new_col", result.column_names)

    def test_add_columns_all_types(self):
        """Test that all column types are correctly added to the schema."""
        settings = MockStructuredOutputSettings()
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.One.name
        
        columns = [
            MockOutputColumn("c1", structured_output.OutputColumnType.String.name),
            MockOutputColumn("c2", structured_output.OutputColumnType.Integer.name),
            MockOutputColumn("c3", structured_output.OutputColumnType.Long.name),
            MockOutputColumn("c4", structured_output.OutputColumnType.Double.name),
            MockOutputColumn("c5", structured_output.OutputColumnType.Boolean.name),
            MockOutputColumn("c6", structured_output.OutputColumnType.String.name, structured_output.OutputColumnQuantity.Multiple.name),
        ]
        settings.output_columns = columns
        
        input_schema = knext.Schema.from_columns([
            knext.Column(knext.int64(), "existing")
        ])
        
        result = structured_output.add_structured_output_columns(input_schema, settings)
        
        self.assertEqual(len(list(result)), 7) # 1 input + 6 output
        self.assertEqual(result["c1"].ktype, knext.string())
        self.assertEqual(result["c2"].ktype, knext.int32())
        self.assertEqual(result["c3"].ktype, knext.int64())
        self.assertEqual(result["c4"].ktype, knext.double())
        self.assertEqual(result["c5"].ktype, knext.bool_())
        self.assertEqual(result["c6"].ktype, knext.list_(knext.string()))

    def test_add_columns_multiple_rows(self):
        settings = MockStructuredOutputSettings()
        settings.target_objects_per_input_row = structured_output.TargetObjectsPerInputRow.Multiple.name
        settings.input_row_id_column_name = "Input Row ID"
        
        field = MockOutputColumn()
        field.name = "new_col"
        field.column_type = structured_output.OutputColumnType.String.name
        settings.output_columns = [field]
        
        input_schema = knext.Schema.from_columns([
            knext.Column(knext.int64(), "existing_col")
        ])
        
        result = structured_output.add_structured_output_columns(input_schema, settings)
        
        # Should add Input Row ID column and new_col
        self.assertEqual(len(list(result)), 3)
        self.assertIn("Input Row ID", result.column_names)
        self.assertIn("new_col", result.column_names)


if __name__ == "__main__":
    unittest.main()
