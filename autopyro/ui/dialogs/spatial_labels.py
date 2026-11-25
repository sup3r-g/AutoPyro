"""Shared labels for spatial interpolation and validation views."""

IDW_REGION_MODES = {'none': 'Обычный IDW', 'clip': 'Обрезка по контуру', 'within': 'Соседи внутри области'}
FORECAST_REASONS = {
    'outside_region': ('Вне выбранной области', '#505050', 'x'),
    'ambiguous_region': ('Граница или перекрытие областей', '#88458b', 'D'),
    'outside_hull': ('Вне обучающей оболочки', '#595959', '^'),
    'outside_regional_hull': ('Вне обучающей оболочки своей области', '#795548', 'v'),
    'insufficient_training': ('Мало обучающих локаций', '#806200', 'P'),
    'insufficient_neighbors': ('Мало соседей в радиусе поиска', '#b35a00', 's'),
    'no_inner_selection': ('Нет общих внутренних прогнозов для подбора', '#633da4', 'X'),
    'method_no_prediction': ('Метод не дал прогноз', '#505050', '+'),
    'unknown': ('Без прогноза', '#555555', 'o'),
}
INTERPOLATION_METHODS = {'kriging': 'Обыкновенный кригинг', 'TIN': 'Линейная интерполяция TIN',
                         'RBF': 'Сглаживаемый RBF', 'nearest': 'Ближайший сосед',
                         'Clough-Tocher': 'Кубическая интерполяция Клафа — Точера',
                         'isotropic-kriging': 'Изотропный кригинг'}
