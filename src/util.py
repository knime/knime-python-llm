import knime.extension as knext
from typing import Callable, List
import re
import pandas as pd


def is_nominal(column: knext.Column) -> bool:
    # Filter nominal columns
    return column.ktype == knext.string() or column.ktype == knext.bool_()


def create_type_filer(ktype: knext.KnimeType) -> Callable[[knext.Column], bool]:
    return lambda c: c.ktype == ktype


ai_icon = "icons/ml.png"

main_category = knext.category(
    path="/community",
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
    expected_type: knext.KnimeType,
    column_purpose: str,
) -> None:
    """
    Raises an InvalidParametersError if a column named column_name is not contained in input_table or has the wrong KnimeType.
    """
    if column_name not in input_table.column_names:
        raise knext.InvalidParametersError(
            f"The {column_purpose} column '{column_name}' is missing in the input table."
        )
    ktype = input_table[column_name].ktype
    if ktype != expected_type:
        raise knext.InvalidParametersError(
            f"The {column_purpose} column '{column_name}' is of type {str(ktype)} but should be of type {str(expected_type)}."
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
        "Output Missing Values",
        "Rows with missing values will not be processed but are included in the output.",
    )
    Fail = (
        "Fail",
        "This node will fail during the execution.",
    )


def skip_missing_values(df: pd.DataFrame, col_name: str, ctx: knext.ExecutionContext):
    # Drops rows with missing values
    df_cleaned = df.dropna(subset=[col_name], how="any")
    n_skipped_rows = len(df) - len(df_cleaned)

    if n_skipped_rows > 0:
        ctx.set_warning(f"{n_skipped_rows} / {len(df)} rows are skipped.")

    return df_cleaned


def handle_missing_and_empty_values(
    df: pd.DataFrame,
    input_column: str,
    missing_value_handling_setting: MissingValueHandlingOptions,
    ctx: knext.ExecutionContext,
):
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
        raise knext.InvalidParametersError(
            f"There are missing values in column {input_column}. See row ID <{missing_row_id}> for the first row that contains a missing value."
        )

    if df.empty:
        raise knext.InvalidParametersError(
            f"""All rows are skipped due to missing values."""
        )

    # Check for empty values
    for id, value in df[input_column].items():
        if not value.strip():
            raise ValueError(
                f"Empty values are not supported. See row ID {id} for the first empty value."
            )

    return df


def output_missing_values(
    df: pd.DataFrame,
    input_column: str,
    missing_value_handling_setting: MissingValueOutputOptions,
    ctx: knext.ExecutionContext,
):
    if df[input_column].isna().all():
        ctx.set_warning("All rows have missing values.")

    has_missing_values = df[input_column].isna().any()

    if has_missing_values:
        if (
            missing_value_handling_setting
            != MissingValueOutputOptions.OutputMissingValues
        ):
            missing_row_id = df[df[input_column].isnull()].index[0]
            raise knext.InvalidParametersError(
                f"There are missing values in column {input_column}. See row ID <{missing_row_id}> for the first row that contains a missing value."
            )

    # Extract non-NaN texts and their indices
    non_nan_mask = df[input_column].notna()
    non_nan_texts = df.loc[non_nan_mask, input_column].tolist()

    # get default numeric indices instead of rowIDs
    df.reset_index(inplace=True)
    indices = df.index[non_nan_mask].tolist()
    return non_nan_texts, indices
