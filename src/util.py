import knime.extension as knext
from typing import Callable


def is_nominal(column):
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


def check_canceled(ctx: knext.ExecutionContext):
    if ctx.is_canceled():
        raise RuntimeError("Execution canceled.")


def pick_default_column(input_table: knext.Schema, ktype: knext.KnimeType) -> str:
    column = next(reversed([c for c in input_table if c.ktype == ktype]), None)
    if column:
        return column.name
    raise knext.InvalidParametersError(
        f"The input table does not contain any columns of type '{str(ktype)}'."
    )


def check_column(
    input_table: knext.Schema,
    column_name: str,
    expected_type: knext.KnimeType,
    column_purpose: str,
):
    """
    Raises an InvalidParametersError if a column named column_name is not contained in input_table or has the wrong KnimeType.
    """
    if not column_name in input_table.column_names:
        raise knext.InvalidParametersError(
            f"The {column_purpose} column '{column_name}' is missing in the input table."
        )
    ktype = input_table[column_name].ktype
    if ktype != expected_type:
        raise knext.InvalidParametersError(
            f"The {column_purpose} column '{column_name}' is of type {str(ktype)} but should be of type {str(expected_type)}."
        )
