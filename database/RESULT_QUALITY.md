# Качество и неопределённость измерений

Миграция `008_result_quality.sql` требует миграции 007. Она добавляет опциональную
таблицу `analysis_result_quality`, связанную с точным результатом по паре
`(analysis_run_id,source_field_id)`. На результат допускается одна текущая запись
метаданных. Это состояние проверки, а не журнал истории изменений.
Исходное значение, тип, единицы, методы, архив и прежние представления сохраняются.

Записи автоматически не создаются: загруженные файлы ещё не были разобраны
на подтверждённые пределы обнаружения и неопределённости. При отсутствии записи
аналитическая выборка показывает `has_quality_metadata=false`,
`qualifier='unreported'`, `review_status='unreviewed'`. Это не признак плохих данных
и не доказательство отсутствия цензурирования.

## Поля

| Поле | Значение |
| --- | --- |
| `qualifier` | `unreported`, `measured`, `below_detection`, `below_quantification`, `less_than`, `greater_than`, `approximate` |
| `source_literal` | Исходная запись из подтверждающего документа, например `<0.1` |
| `reporting_bound` | Числовая граница для `less_than`/`greater_than`; обязательна для них |
| `detection_limit` / `quantification_limit` | LOD/LOQ в исходных единицах результата |
| `absolute_uncertainty` | Абсолютная неопределённость в исходных единицах результата |
| `uncertainty_kind` | `standard`, `expanded` или `unspecified`; обязательно при заданной неопределённости |
| `coverage_factor` | Коэффициент охвата для расширенной неопределённости, если известен |
| `confidence_level` | Уровень доверия в долях, строго между 0 и 1, если известен |
| `review_status` | `unreviewed`, `accepted`, `suspect`, `rejected` |
| `reviewed_by` / `reviewed_at` | Кто и когда проверил; обязательны для статуса, отличного от `unreviewed` |
| `evidence` | Обязательное описание источника этих сведений: файл, документ, лист/строка или лабораторный протокол |
| `note` | Пояснение |

LOD/LOQ и неопределённость должны быть конечными неотрицательными числами,
LOQ не ниже LOD, если заданы оба. Граница сравнения может быть отрицательной.
Для `below_detection` требуется LOD, для `below_quantification` — LOQ.
Числовые квалификаторы и числовые характеристики запрещены для текстовых результатов;
статус экспертной проверки у текстового результата допустим.
Ни единицы, ни уровни доверия автоматически не преобразуются.

Квалификатор описывает смысл исходного результата; он не заменяет исходное число
границей, нулём или половиной LOD. Несогласованность исходного числа с границей
не отклоняется автоматически, поскольку источник мог использовать условный код.
Решение фиксируют в статусе проверки и пояснении.

## Представления

- `v_geochemical_measurement_qc`: все столбцы `v_geochemical_measurement`
  плюс метаданные качества. В выборке остаются все результаты, в том числе rejected.
- `v_geochemical_quality_summary`: счётчики по импорту, группе и исходному полю:
  аннотированные, непроверенные, принятые, подозрительные, отклонённые,
  цензурированные и с известной неопределённостью. `censored_count` включает
  below_detection, below_quantification, less_than и greater_than.

Оба представления и таблица доступны роли `readonly` для SELECT.
Приложение продолжает работать со старыми представлениями; фильтрация rejected
или использование неопределённостей включается явно в новых запросах.

```sql
-- Данные конкретной пробы со статусом качества:
SELECT sample_id,wellbore_name,top_md,base_md,group_name,repeat_number,
       source_label,value_numeric,unit_label,qualifier,detection_limit,
       absolute_uncertainty,uncertainty_kind,review_status,quality_evidence
FROM geochem.v_geochemical_measurement_qc
WHERE sample_id=1
ORDER BY analysis_run_id,source_field_id;

-- Только явно принятые, нецензурированные числовые измерения:
SELECT * FROM geochem.v_geochemical_measurement_qc
WHERE review_status='accepted' AND qualifier='measured'
  AND value_numeric IS NOT NULL;

-- Покрытие метаданными качества:
SELECT group_name,sum(result_count) AS results,sum(annotated_count) AS annotated,
       sum(unreviewed_count) AS unreviewed,sum(uncertainty_count) AS with_uncertainty
FROM geochem.v_geochemical_quality_summary GROUP BY group_name;
```

Для ручного заполнения admin использует INSERT с существующими идентификаторами
результата и доказательством из источника. Не следует массово выставлять
`measured`/`accepted` по факту наличия числового значения. После изменения
метаданные сразу видны в новых представлениях, повторный импорт не требуется.

## Применение

```powershell
.venv/Scripts/python.exe -m autopyro.database.tools.validate_result_quality
.venv/Scripts/python.exe -m autopyro.database.tools.validate_result_quality --apply
```

Проверка сравнивает агрегированные отпечатки существующих таблиц, число результатов,
счётчики сводки, неизвестные характеристики, допустимую аннотацию и недопустимые
метаданные (отрицательные/неконечные пределы, противоречащие друг другу LOD/LOQ,
неполная неопределённость, неподтверждённый статус, текстовые значения,
несуществующие и дублирующие ключи). Тестовые аннотации откатываются также
при `--apply`. Первый запуск откатывает и DDL, второй фиксирует DDL после проверок.
Применять без параллельного изменения данных.
