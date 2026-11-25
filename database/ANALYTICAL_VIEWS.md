# Аналитические представления

Миграция `007_analytical_views.sql` требует миграций 005 и 006. Она добавляет
обычные SQL-представления без изменения таблиц и старых интерфейсов приложения.
Представления всегда читают актуальные данные; обновлять их вручную не нужно.
Все четыре доступны роли `readonly`.

| Представление | Одна строка | Назначение |
| --- | --- | --- |
| `v_geochemical_measurement` | Результат поля в конкретном запуске | Универсальная выборка по пробе, скважине, глубине, показателю, методу и источнику |
| `v_geochemical_run` | Запуск анализа | Все его результаты в JSON, ключ — `source_field_id` |
| `v_pyrolysis_report` | Запуск группы `Pyrol%` | Столбцы TOC, S1, S2, S3, Tmax, HI, OI |
| `v_geochemical_availability` | Ствол/исходная скважина/импорт/поле/группа/система глубин | Количество результатов, проб и запусков, диапазон известных глубин |

Длинное представление сохраняет все типизированные значения, исходные названия
и единицы, идентификаторы метода и показателя, признаки проверки справочников,
номер повтора, запуск, файл, SHA256 и номер исходной строки. `value_numeric`
позволяет одинаково фильтровать целые и вещественные значения; исходные
`value_int` и `value_float` сохраняются. Это не преобразование единиц.
Поля текстовых метаданных также остаются в представлении.

`well_name` — имя исходной скважины, `wellbore_name` — имя в общем справочнике.
Площадь и координаты берутся из исходной записи `well`. Пустые скважины/глубины
не исключают результаты. Пробы без результатов отсутствуют в длинной выборке;
все пробы, включая такие, доступны через `v_sample_wellbore`.

`depth_interval_valid` означает, что обе MD-глубины конечны, неотрицательны
и подошва не выше кровли. Точечный интервал допустим. Признак не подтверждает
datum. В сводку глубин входят только такие интервалы; результатам с неизвестными
или некорректными глубинами всё равно соответствует строка и счётчик.
MD/TVD и datum литостратиграфии не выводятся из числового совпадения глубин.

В пиролитической таблице используются точные исходные названия с единицами
из каталога; отсутствие соответствующего поля/результата даёт NULL.
Каждый запуск и повтор сохраняется отдельно. MAX в развороте выбирает
единственный результат поля внутри запуска (уникальность обеспечена схемой),
а не максимум по независимым измерениям. Канонические переназначения справочников
не меняют фиксированный разворот исходных полей. Он не объединяет методики.

## Примеры

```sql
-- Полный набор данных для одной пробы:
SELECT * FROM geochem.v_geochemical_measurement
WHERE sample_id=1
ORDER BY analysis_run_id,source_field_id;

-- Пиролиз по скважине в заданном диапазоне MD (пересечение замкнутых интервалов):
SELECT sample_id,well_name,top_md,base_md,depth_datum,
       group_name,repeat_number,analysis_run_id,
       toc_wt_pct,s1_mg_g,s2_mg_g,tmax_c
FROM geochem.v_pyrolysis_report
WHERE well_name='7221/4-1'
  AND top_md<=2000 AND base_md>=1000
  AND top_md>=0 AND base_md>=top_md
  AND top_md NOT IN ('NaN'::float8,'Infinity'::float8,'-Infinity'::float8)
  AND base_md NOT IN ('NaN'::float8,'Infinity'::float8,'-Infinity'::float8)
ORDER BY top_md,analysis_run_id;

-- TOC по площади; методики, повторы и единицы видны в каждой строке:
SELECT sample_id,wellbore_id,wellbore_name,top_md,base_md,depth_datum,
       group_name,repeat_number,analysis_run_id,value_numeric,unit_label,
       mapping_reviewed
FROM geochem.v_geochemical_measurement
WHERE source_label='TOC [%wt]' AND area='Barents Sea'
ORDER BY wellbore_id,top_md,analysis_run_id;

-- Найти стабильные идентификаторы для фильтрации вместо названий:
SELECT source_field_id,group_name,source_label,indicator_id,indicator_name,
       method_name,unit_label,sum(result_count) AS results
FROM geochem.v_geochemical_availability
GROUP BY source_field_id,group_name,source_label,indicator_id,indicator_name,
         method_name,unit_label
ORDER BY results DESC;

-- Все результаты запуска одним объектом JSON:
SELECT analysis_run_id,sample_id,group_name,repeat_number,values_by_field
FROM geochem.v_geochemical_run
WHERE sample_id=1 ORDER BY analysis_run_id;
```

При выборке по общему стволу используйте `wellbore_id`, по исходной скважине —
`well_id`. Название пробы не уникально. Один показатель может иметь несколько
результатов на пробу; не сводите их к одному значению без правила выбора метода
и повтора. Сводка доступности намеренно не рассчитывает общие средние:
сопоставимость измерений должна быть проверена отдельно.

## Установка и проверка

```powershell
.venv/Scripts/python.exe -m autopyro.database.tools.validate_analytical_views
.venv/Scripts/python.exe -m autopyro.database.tools.validate_analytical_views --apply
```

Первый запуск проверяет с откатом, второй фиксирует после проверок. Проверяются
отпечатки всех таблиц, повторный запуск, сохранение каждого типизированного
результата, число запусков, точность JSON и семи столбцов пиролиза, счётчики
доступности и права readonly. Применять без параллельного изменения данных.
Существующие импортёры не требуют изменений: представления читают их таблицы.

Полная сводка и выгрузка JSON по всей базе требуют агрегации. Для интерактивной
работы сначала ограничивайте выборку идентификатором пробы/скважины/запуска.
Материализация и новые индексы добавляются только после измерения конкретных
запросов через EXPLAIN (ANALYZE, BUFFERS).
