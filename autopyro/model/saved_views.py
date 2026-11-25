"""Maintain named table views independently of their Qt editors."""

from copy import deepcopy


def validate_view_name(view, name, previous=None):
    name = name.strip()
    if not name:
        raise ValueError('Введите название представления')
    if name != previous and name in view.get('named', {}):
        raise ValueError('Представление «{0}» уже существует'.format(name))
    return name


def create_view(view, name, settings):
    name = validate_view_name(view, name)
    view.setdefault('named', {})[name] = deepcopy(settings)
    return name


def update_view(view, name, changes):
    named = view.get('named', {})
    if name not in named:
        raise ValueError('Сохранённое представление отсутствует')
    updated = {**named[name], **deepcopy(changes)}
    if updated == named[name]:
        return False
    named[name] = updated
    return True


def rename_view(view, old_name, name):
    named = view.get('named', {})
    if old_name not in named:
        raise ValueError('Сохранённое представление отсутствует')
    name = validate_view_name(view, name, old_name)
    if name != old_name:
        view['named'] = {name if key == old_name else key: value for key, value in named.items()}
    return name


def delete_view(view, name):
    named = view.get('named', {})
    if name not in named:
        raise ValueError('Сохранённое представление отсутствует')
    del named[name]
