import knime.extension as knext


def is_nominal(column):
    # Filter nominal columns
    return column.ktype == knext.string() or column.ktype == knext.bool_()