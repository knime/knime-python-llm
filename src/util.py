import knime.extension as knext
from typing import Callable, List
import re


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
    ctx: knext.ConfigurationContext,
    input_table_spec: knext.Schema,
    column_name: str,
):
    """
    If the output column name collides with an input column name, it's been made unique by appending (#<count>).
    For example, if "column" exists as an input column name and is entered as an output column name,
    the output column name will be "column (#1)". Adding "column" or "column (#1)" again will result
    in "column1 (#2)".
    """
    basename = column_name.strip()

    existing_column_names = set(input_table_spec.column_names)

    if column_name not in existing_column_names:
        return column_name

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
