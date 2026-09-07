"""Shared business-attribute exclusion rules for all prediction methods."""


COMMON_EXCLUDED_ATTRIBUTE_COLUMNS = (
    'case',
    'event',
    'time',
    'step_index',
    'id',
    'EventID',
    'OfferID',
    'case:REG_DATE',
)


def is_excluded_attribute(column, additional_columns=()):
    """Return True for control/identifier fields and all Variant fields."""
    column_lower = str(column).lower()
    excluded_lower = {
        str(name).lower()
        for name in (*COMMON_EXCLUDED_ATTRIBUTE_COLUMNS, *additional_columns)
    }
    return column_lower in excluded_lower or 'variant' in column_lower
