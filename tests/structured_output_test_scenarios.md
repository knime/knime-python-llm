# Test Scenarios for LLM Prompter Structured Output Integration

This document outlines the combined test scenarios for the integration tests of the structured output functionality in the LLM Prompter nodes.

## 1. Comprehensive Single Object Extraction (Happy Path)
- **Goal**: Verify successful extraction of multiple data types, handling of long integers, column name collisions, sanitization of special characters, and the influence of the target object description.
- **Input**: A table with a column named `Category` (to test collision). 
    - **Prompt**: "Extract the following from this text: 'The user John Doe (ID: 9223372036854775807) is 35 years old, has a balance of 1500.75 and is currently active. The overall category is Finance.'"
- **Configuration**:
    - **Target Object Name**: "User-Profile Data!" (to test sanitization)
    - **Target Object Description**: "Extracted information about a user and their financial status."
    - **Target Objects Per Input Row**: `One`
    - **Output Columns**:
        - `Name` (String)
        - "Full ID?" (Long) (to test sanitization and precision)
        - `Age` (Integer)
        - `Balance` (Double)
        - `IsActive` (Boolean)
        - `Category` (String) (to test name collision)

### Input CSV
```csv
Category,Prompt
General,"Extract the following from this text: 'The user John Doe (ID: 9223372036854775807) is 35 years old, has a balance of 1500.75 and is currently active. The overall category is Finance.'"
```

### Expected Output CSV
```csv
Category,Prompt,Name,"Full ID?",Age,Balance,IsActive,"Category (out)"
General,"Extract the following from this text: 'The user John Doe (ID: 9223372036854775807) is 35 years old, has a balance of 1500.75 and is currently active. The overall category is Finance.'",John Doe,9223372036854775807,35,1500.75,true,Finance
```

## 2. Advanced Multiple Objects, List Columns, and Missing Values
- **Goal**: Verify that the node handles multiple objects per row, list columns within those objects, and missing information simultaneously.
- **Input**: A table with one row: "Extract each student, their grades, and their hometown if mentioned: 'Alice got A and B, lives in London. Bob got C, D, and E (hometown unknown). Charlie is mentioned but has no grades and no city.'" RowID: `Row0`.
- **Configuration**:
    - **Target Objects Per Input Row**: `Multiple`
    - **Input Row ID Column Name**: `Original Row ID`
    - **Output Columns**:
        - `Student` (String)
        - `Grades` (String, Quantity: `Multiple`)
        - `Hometown` (String)

### Input CSV
```csv
Prompt
"Extract each student, their grades, and their hometown if mentioned: 'Alice got A and B, lives in London. Bob got C, D, and E (hometown unknown). Charlie is mentioned but has no grades and no city.'"
```

### Expected Output CSV
```csv
RowID,Prompt,"Original Row ID",Student,Grades,Hometown
Row0_0,"Extract each student, their grades, and their hometown if mentioned: 'Alice got A and B, lives in London. Bob got C, D, and E (hometown unknown). Charlie is mentioned but has no grades and no city.'",Row0,Alice,"[""A"", ""B""]",London
Row0_1,"Extract each student, their grades, and their hometown if mentioned: 'Alice got A and B, lives in London. Bob got C, D, and E (hometown unknown). Charlie is mentioned but has no grades and no city.'",Row0,Bob,"[""C"", ""D"", ""E""]",
Row0_2,"Extract each student, their grades, and their hometown if mentioned: 'Alice got A and B, lives in London. Bob got C, D, and E (hometown unknown). Charlie is mentioned but has no grades and no city.'",Row0,Charlie,,
```

## 3. Batch Processing and Empty Results
- **Goal**: Ensure the node processes multiple input rows independently and handles rows that yield zero extracted objects.
- **Input**: A table with 3 rows:
    - **Row 0**: "I like apples and bananas."
    - **Row 1**: "I prefer eating pizza."
    - **Row 2**: "I like oranges."
- **Configuration**:
    - **Target Objects Per Input Row**: `Multiple`
    - **Output Columns**:
        - `Fruit` (String)

### Input CSV
```csv
Prompt
"I like apples and bananas."
"I prefer eating pizza."
"I like oranges."
```

### Expected Output CSV
```csv
RowID,Prompt,"Original Row ID",Fruit
Row0_0,"I like apples and bananas.",Row0,Apple
Row0_1,"I like apples and bananas.",Row0,Banana
Row2_0,"I like oranges.",Row2,Orange
```

## 4. Error Handling: Model Incompatibility
- **Goal**: Verify that the node fails correctly when configured with a model that does not support structured output.
- **Setup**: Use a model that defines `supported_output_formats` without `OutputFormatOptions.Structured`.
- **Expected Result**:
    - Configuration error: "The selected output format 'Structured' is not supported by the LLM."
