# Полный дамп geochemistry

Создан 24 октября 2026 года с PostgreSQL 18.6 командой pg_dump --format=custom.

## Состав

- geochemistry.dump — сжатый архив всей базы: схема, данные, исходные JSON-снимки, индексы, ограничения, представления, последовательности, владельцы и права объектов.
- SHA256SUMS.txt — SHA-256 архива для проверки целостности.
- verification.json — результат проверки восстановления и контрольные суммы всех таблиц.

Размер архива: 22 601 107 байт (около 21,55 MiB).
SHA-256: a1e473a064f3c7ea04ba568d9646ed75469d16f8ff7a0a85f7ebca318bfce164.

В архив входят геохимическая база и литостратиграфия из Excel: 14 таблиц, 3 представления, 11 последовательностей. В том числе 31 665 проб, 80 495 анализов, 1 056 734 результата и 2 514 литостратиграфических интервалов.

## Проверка

Архив восстановлен через pg_restore --exit-on-error --single-transaction в отдельную пустую базу.
С исходной базой совпали количества строк и контрольные суммы полного содержимого всех 14 таблиц, состояния 11 последовательностей, определения 3 представлений и 104 ограничений.
Исходная база не изменялась. Проверочная база удалена после успешного сравнения.

## Восстановление в новую базу через текущий Compose

Команды из корня проекта D:\\Code\\Python\\AutoPyro. Имя geochemistry_restored должно быть свободным.

```powershell
docker compose cp 'database/backups/geochemistry_2026-10-24_3efe0b53/geochemistry.dump' postgres:/tmp/geochemistry_restore.dump
docker compose exec -T postgres createdb -U geochem --template=template0 --owner=geochem geochemistry_restored
docker compose exec -T postgres pg_restore --exit-on-error --single-transaction -U geochem --dbname=geochemistry_restored /tmp/geochemistry_restore.dump
```

Для другого сервера используйте PostgreSQL 18 или совместимую более новую версию и роль geochem, созданную заранее. pg_restore применяется к архиву custom; psql напрямую этот формат не читает.

Это дамп отдельной базы. Определения серверных ролей, пароли и файлы конфигурации PostgreSQL в него не входят. В текущем Compose роль geochem уже существует.

Документация формата: https://www.postgresql.org/docs/18/app-pgdump.html

