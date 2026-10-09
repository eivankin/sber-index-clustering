# Земля под тратами

Муниципальные округи, разделённые не по объёму трат, а по тому, чем место устроено: классы земли по
снимкам, ночные огни, вегетация, высота — вместе с формой трат по категориям, но без самой суммы.

**Что смотреть**

Живая версия — GitHub Pages по адресу <https://eivankin.github.io/sber-index-clustering/>; он заработает
после того, как Pages будет включён. До этого смотрят прямо в файлы репозитория:

- Интерактивная презентация — [`web/presentation/index.html`](web/presentation/index.html). Пять вкладок:
  о проекте, откуда данные, как часто можно пересчитывать, как выбраны признаки и сколько типов, карта.
- Полное описание методологии — [`docs/report/methodology-ru.md`](docs/report/methodology-ru.md);
  на Pages она лежит по адресу [`web/report/methodology-ru.md`](web/report/methodology-ru.md).
- Интерактивная карта всей страны — [`web/map/full/index.html`](web/map/full/index.html).

Pages публикуется из ветки `submission-2026`, корень публикации — `web/`. В этом каталоге лежит
`web/index.html`, который ведёт на презентацию, и `web/.nojekyll`: без него Jekyll не положит в сайт
файл геометрии `web/map/map_geom.bin`, а карта без него не открывается. Карта грузит 71,8 МБ геометрии
при первом открытии вкладки, поэтому на медленном интернете она появляется позже остальных вкладок.

## О чём результат, коротко

- **2 660 муниципалитетов** собраны из OpenStreetMap и склеены с официальными площадями Росстата;
  кластер присвоен **2 016** из них, **644** остались без цвета — по ним нет ни одной строки расходов
  либо они не прошли порог покрытия (`artifacts/geo/geometry_diagnostics.json`,
  `artifacts/landing/territory_measures.json`).
- **Пространство признаков — 13 столбцов:** пять классов Dynamic World, ночные огни, вегетация MODIS,
  высота местности и пять долей трат по категориям. **Сумма трат в него не входит** — это записанное
  решение (`artifacts/validation/k_decision.json`).
- **Почему сумма снаружи:** измерено, что она забирает **22,6 %** разделения между центрами кластеров,
  когда находится внутри, и доля земли падает с 71,4 % до 56,4 %
  (`artifacts/validation/amount_dominance.json`). Между разбиением с суммой и присланным ответом
  переезжают **87,75 %** округов — то есть сумма и есть ответ «богатые против бедных»
  (`artifacts/landing/amount_comparison_ramp_note.json`).
- **Почему ровно пять типов:** правило из пяти требований к осмысленности было записано до сравнений
  (`artifacts/validation/meaningfulness_rule.json`). По батареям ICVI ответ не победитель: лучший
  силуэт даёт k=2, лучший суммарный ранг k=14 (`artifacts/validation/k_sweep.csv`,
  `artifacts/validation/k_sweep_summary.json`). k=5 выбран потому, что только он даёт пятый тип —
  лесную глубинку: 283 округа при средней долготе около 99°, лес 0,288, огни 0,633 и самая низкая
  плотность трат на квадратный километр при сумме, которая выглядит средней
  (`artifacts/interpretation/cluster_names.json`, `artifacts/interpretation/cluster_profile.csv`).
- **Правило ребра:** `w_ij = f(d_ij) · supply_i · demand_j`, направление только «производитель →
  потребитель», расстояние — дорожная матрица SberIndex, а не прямая между центрами
  (`configs/edge_rules.yaml`). Его влияние на кластеризацию измерено: признаки сети, добавленные в
  пространство, роняют совпадение с базовым разбиением до **ARI 0,1342 / NMI 0,1219**, и переезжают
  **81,69 %** округов — поэтому признаки рёбер в пространство не вошли
  (`artifacts/validation/edge_influence.json`).
- **Движение между 2023 и 2024:** из 13 столбцов 2024 года имеют 11 — пять классов земли, вегетация и
  пять долей трат; ночные огни заморожены записанным решением, высота не пересчитывается. На этом
  пересчёте **106 округов (5,26 %)** меняют тип, ARI 0,8783, NMI 0,8679
  (`artifacts/validation/partition_2024.json`). Если пересчитывать только землю — 75 округов (3,72 %),
  только вегетацию — 24 (1,19 %). Замена ночных огней 2024 годом двигает **44,79 %** — именно поэтому
  они заморожены (`artifacts/validation/refresh_2024_ndvi_viirs.json`).
- **Что не утверждается:** годовая потеря лесного покрова по Dynamic World не подтверждена независимым
  годовым сигналом; 546 гипотез перехода застройки разошлись как 8 подтверждений, 1 опровержение и 537
  неразрешимых; лицензия Hansen не прочитана ни с одной страницы; более частые заборы земли не
  измерялись (`artifacts/dynamics/geospatial_consistency.json`,
  `artifacts/geo/added_layers_measures.json`, `artifacts/validation/dynamic_world_resolution.json`).

## Как это устроено в репозитории

| Что | Где |
| --- | --- |
| Библиотека (кластеризация, ICVI, сеть, кэш Earth Engine) | `src/sberindex/` |
| Гиперпараметры и правила | `configs/*.yaml` |
| Сборка признакового пространства и разбиения | `scripts/build_features.py`, `scripts/build_panel.py`, `scripts/run_clustering.py` |
| Обоснование k и батарея ICVI | `scripts/sweep_k_interpretability.py`, `scripts/check_icvi.py` |
| Правила рёбер и их влияние | `scripts/build_edges.py`, `scripts/measure_edge_influence.py`, `scripts/check_edges.py` |
| Движение 2023 → 2024 | `scripts/build_spending_2024_shares.py`, `scripts/build_2024_partition.py` |
| Имена и профиль типов | `scripts/build_cluster_names.py`, `scripts/interpret_clusters.py` |
| Презентация и её картинки | `scripts/build_stage_one_presentation.py`, `scripts/build_vector_choropleth.py`, `scripts/build_spending_vs_structure.py` |
| Карта | `scripts/build_map_geometry_bin.py`, `scripts/build_map_full_page.py`, `scripts/build_map_data.py` |
| Частота обновления входов | `scripts/build_cadence_families.py`, `scripts/build_cadence_story.py` |
| Публичные числа и их происхождение | `scripts/build_claims.py` → `artifacts/landing/claims.json` |
| Контролёры согласованности | `scripts/check_*.py`, общий запуск `scripts/check_all.py` |

## Как запустить

```bash
python -m venv .venv && .venv/bin/python -m pip install -e .
PYTHONPATH=src .venv/bin/python scripts/build_features.py
PYTHONPATH=src .venv/bin/python scripts/run_clustering.py
PYTHONPATH=src .venv/bin/python scripts/sweep_k_interpretability.py
PYTHONPATH=src .venv/bin/python scripts/build_2024_partition.py
PYTHONPATH=src .venv/bin/python scripts/build_stage_one_presentation.py
PYTHONPATH=src .venv/bin/python scripts/build_map_geometry_bin.py
PYTHONPATH=src .venv/bin/python scripts/build_map_full_page.py
PYTHONPATH=src .venv/bin/python scripts/check_all.py
```

Строки SberIndex и таблица Росстата в этот репозиторий не входят: это входные данные, а не результат.
Без них первые два шага не запускаются; они берутся из официальных источников и кладутся в `data/`
локально. Ответы Earth Engine кэшируются **вне репозитория**, и воспроизводимость означает пересборку из
кэша, а не повторную загрузку: ключ кэша записывает, какой актив и какая геометрия дали ответ
(`src/sberindex/geecache.py`). Пустой ответ не может притвориться измерением.

## Лицензии данных

- Траты — **SberIndex**, CC BY-SA 4.0.
- Базовая карта — **OpenStreetMap** contributors, ODbL.
- Классы земли — Google **Dynamic World** v1; ночные огни — **VIIRS** DNB; вегетация — **MODIS** MOD13Q1;
  высота — **SRTM** и **Copernicus GLO30**; площадь — официальный реестр **Росстата**.

## Ветка и раздача

Эта ветка содержит только результат: презентацию, карту, отчёт, конфиги, код и те артефакты, на которые
ссылается отчёт.
