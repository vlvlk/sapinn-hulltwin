# PINN-HullTwin — «Живой корпус»

Гидродинамический цифровой двойник судна с самодиагностикой обрастания: PINN решает
обратную задачу и восстанавливает поле шероховатости корпуса $k_s(x)$ и деградацию винта
из судовых логов, далее — калиброванная модель сопротивления и оптимизация режима движения.

## Стек

- Python **3.14**, менеджер пакетов — **uv**
- DL-ядро: PyTorch 2.10 + torchdiffeq (вариант A) или JAX + Equinox + Diffrax (вариант B)
- Физическое ядро MVP: numpy/scipy (аналитика + нейронный ODE Кармана)
- Качество: ruff, mypy, pytest

## Структура

```
plans/          # архитектура MVP
src/hulltwin/
  physics/      # карман, ITTC-1957 + Bowden-Davison, K_T/K_Q винта, волновое сопротивление
  pinn/         # PINN-наблюдатель (v0 — заготовка)
  data/         # генератор синтетики, адаптеры внешних датасетов (Shifts, ERA5/Copernicus)
tests/          # unit-тесты физики против аналитики
notebooks/      # демо: идентифицируемость k_s, прогноз мощности
```

## Данные

- `data/external/power_consumption_upload/` — Shifts / Marine Cargo Vessel
  Power Consumption (Zenodo 7057666, CC BY-NC-SA 4.0). Датасет в репозиторий
  **не коммитится**: `train.csv` (~109 MB) превышает лимит GitHub в 100 MB,
  а весь каталог `data/external/` исключён через `.gitignore`. Полный архив
  скачивается с Zenodo: `uv run python scripts/download_shifts.py`.
  Известный инцидент: во время разработки MVP zenodo.org целиком лежал
  (HTTP 504 на все эндпоинты в течение нескольких часов) — скрипт
  ретраит временные 5xx с экспоненциальной задержкой, но при полной
  недоступности нужно дождаться восстановления или развернуть датасет
  из локального бэкап-архива.
- Адаптер: `hulltwin.data.shifts` — приведение к схеме двойника
  (STW, волны, ветер/течение, возраст сухого дока -> k_s(t)) и
  физически структурированная модель мощности (V^3 + ITTC-1957 с
  надбавкой за шероховатость + волновое слагаемое; 4 калибруемых
  скаляра, NNLS).

## Быстрый старт

```bash
uv sync
uv run pytest
```

## Дорожная карта MVP

1. [x] Скелет репо
2. [x] Физическое ядро + тесты (`hulltwin.physics`)
3. [x] Генератор синтетических рейсов (`hulltwin.data.synthetic`)
4. [x] PINN-наблюдатель v0 + анализ идентифицируемости (`hulltwin.pinn.observer`)
5. [x] Адаптер Shifts power-consumption dataset (`hulltwin.data.shifts`, `scripts/shifts_benchmark.py`)
6. [x] Отчёт-дашборд (`hulltwin.report`, `scripts/generate_report.py`)
7. [x] Демо-ноутбуки (`notebooks/01_synthetic_id.ipynb`, `notebooks/02_shifts_power.ipynb`)
8. [x] CI (GitHub Actions: ruff + mypy + pytest + smoke report/notebooks)

## Демо

```bash
uv run python scripts/generate_report.py          # отчёт -> reports/hulltwin_report.png
uv run python scripts/generate_report.py --days 720 --epochs 4000
uv run python scripts/shifts_benchmark.py         # Shifts power-consumption бенчмарк

# демо-ноутбуки (для ноутбука 02 сначала: uv run python scripts/download_shifts.py):
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_synthetic_id.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_shifts_power.ipynb
# либо интерактивно: uv run jupyter lab notebooks/
```

## CI

GitHub Actions (`.github/workflows/ci.yml`), два джоба:
- **quality**: `ruff check` + `mypy src scripts tests` + `pytest`;
- **smoke**: полный прогон пайплайна с укороченным обучением
  (`generate_report.py --days 180 --epochs 500`) и headless-исполнение
  обоих ноутбуков через `nbconvert` (без Shifts-данных ноутбук 02
  корректно пропускает вычислительные ячейки).

Результат на Shifts (физически структурированная модель, 4 калибруемых
скаляра, train 531k записей): медианная ошибка мощности 7.4% на dev_in
и 13.8% на сдвинутом dev_out — физическая структура (V^3 + ITTC +
шероховатость + волны) смягчает деградацию под сдвигом по сравнению с
чёрным ящиком, но полностью её не устраняет (направление будущих улучшений).

Результат на синтетике (720 дней, 2 очистки): corr(k_s траектории) = 0.95,
ошибка итоговой деградации винта k_p = 0.25%, медианная ошибка мощности 5.4%.

## Идентифицируемость (важно)

Из судовых логов ship-масштаба данные содержат ровно одно уравнение на день
для корпуса (тождество тяги) — наблюдаемо только Cf-взвешенное среднее
k_s_mean(t). Пространственный профиль k_s(x) восстанавливается только как
диагностика с априорной формой (корма зарастает быстрее); идентификация
градиента требует полосовых данных — см. docstring `hulltwin.pinn.observer`.
