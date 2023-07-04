import knime.extension as knext


def is_nominal(column):
    # Filter nominal columns
    return column.ktype == knext.string() or column.ktype == knext.bool_()


main_category = knext.category(
    path="/community",
    level_id="kai",
    name="AI",
    description="",
    icon="icons/ml.png",
)
