# Tests

This directory contains tests for the KNIME Python LLM extension.

## Running Tests

The tests use pytest and require the full project environment with all dependencies installed.

### Using pixi (recommended)

```bash
pixi run test
```

### Using unittest directly

If you have all dependencies installed in your Python environment:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### Using pytest directly

If pytest is installed in your environment:

```bash
pytest tests/
```

## Test Structure

- `conftest.py` - Common test configuration that ensures proper import paths
- `test_utils.py` - Tests for utility functions in `src/util.py`
- `test_structured_output.py` - Comprehensive tests for the structured_output module

## Test Coverage

### structured_output Module

The test suite for `src/models/structured_output.py` includes:

- **TestOutputFieldType** - Tests for enum values
- **TestValidateOutputFields** - Tests for field validation logic
- **TestCreatePydanticModel** - Tests for Pydantic model creation (both One and Many modes)
- **TestGetOutputFieldKnimeType** - Tests for KNIME type mapping
- **TestGetOutputFieldPyArrowType** - Tests for PyArrow type mapping
- **TestMakeRowIdsUnique** - Tests for row ID uniqueness after explosion
- **TestExplodeLists** - Tests for list column explosion
- **TestCreateEmpty** - Tests for empty table creation
- **TestStructuredResponsesToTable** - Tests for converting Pydantic responses to tables
- **TestPostprocessTable** - Tests for table postprocessing
- **TestAddStructuredOutputColumns** - Tests for schema modification

## Writing New Tests

Follow the pattern from knime-python-extension-template:

1. Use unittest.TestCase for test classes
2. Import modules from `src.*` (conftest.py handles the path setup)
3. Use KNIME's testing utilities from `knime.extension.testing` when available
4. Keep tests focused and test one thing at a time
