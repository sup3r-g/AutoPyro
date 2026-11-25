"""Maintain named sample memberships and their saved view references."""


def create_sample_set(view, name, row_ids):
    name = name.strip()
    sets = view.get("sample_sets", {})
    if not name or name in sets:
        raise ValueError('Введите уникальное название набора.')
    view.setdefault("sample_sets", {})[name] = row_ids


def rename_sample_set(view, old_name, name):
    sets = view.get("sample_sets", {})
    if old_name not in sets:
        raise ValueError('Сохранённый набор проб отсутствует.')
    name = name.strip()
    if name == old_name:
        return name
    if not name or name in sets:
        raise ValueError('Введите уникальное название набора.')
    view["sample_sets"] = {name if key == old_name else key: ids for key, ids in sets.items()}
    for settings in (view.get("current", {}), *view.get("named", {}).values()):
        if settings.get("sample_set") == old_name:
            settings["sample_set"] = name
    return name


def check_sample_set_deletion(view, name):
    linked = [label for label, settings in view.get("named", {}).items()
              if settings.get("sample_set") == name]
    if linked:
        raise ValueError(
            'Набор проб используется в представлениях: {0}. Сначала измените или удалите эти представления.'.format(
                ', '.join(linked),
            )
        )


def delete_sample_set(view, name):
    sets = view.get("sample_sets", {})
    if name not in sets:
        raise ValueError('Сохранённый набор проб отсутствует.')
    check_sample_set_deletion(view, name)
    del sets[name]
    current = view.get("current", {})
    if current.get("sample_set") == name:
        current.pop("sample_set")
