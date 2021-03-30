from ._app import App


def ordered_field_names(app: App, display=None, min_count=1, fill_with: str or None = None):
    field_names = list(app.fieldnames)  # in order
    if isinstance(display, str) and display in field_names:
        field_names.remove(display)
        field_names.insert(0, display)
    elif isinstance(display, (tuple, list)):
        for display_name in reversed(display):
            if display_name in field_names:
                field_names.remove(display_name)
                field_names.insert(0, display_name)
    return field_names + [fill_with] * max(0, min_count - len(field_names))
