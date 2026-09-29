# Build the Doom learning dashboard in Power BI Desktop

This guide assumes no previous Power BI experience. The dashboard uses generated
gameplay evidence, not training accuracy, so it is useful both for debugging and
for explaining the project in a job interview.

## 1. Create the CSV tables

From the repository root, run:

```powershell
python examples/doom/export_power_bi.py `
  --runs runs/movingman-episode1-full `
  --output power-bi-export
```

The command creates five files:

- `attempts.csv`: one row per mission attempt and its outcome.
- `steps.csv`: one row per controller decision.
- `pickups.csv`: first-seen visible/map-known opportunities and confirmed events.
  Older traces whose old schema did not prove camera visibility are explicitly
  marked `legacy_weapon_observation_unverified`.
- `route_transitions.csv`: sector-to-sector movement.
- `judge_findings.csv`: causal judge failures and their steps.

The generated folder is local analysis output. Do not commit it unless you have
checked its size and removed machine-specific or private information.

## 2. Load the data

1. Install and open **Power BI Desktop**.
2. Select **Get data → Text/CSV**.
3. Import each of the five CSV files.
4. Select **Transform Data** before loading.
5. In Power Query, set identifiers such as `run_id`, `map`, `mode`, `objective`
   and `name` to **Text**. Set `step`, `tic`, `kills` and sector columns to
   **Whole number**. Set health, positions, distance and `wall_seconds` to
   **Decimal number**. Set completed/dead/timed_out/confirmed to **True/False**.
6. Select **Close & Apply**.

## 3. Create relationships

Power BI needs a unique Runs table because `run_id` repeats in the event tables.
Choose **Modeling → New table** and enter:

```DAX
Runs = DISTINCT(attempts[run_id])
```

In Model view, connect `Runs[run_id]` one-to-many to the `run_id` column in all
five imported tables. Use single-direction filtering from Runs to each table.

## 4. Add useful measures

Choose **Modeling → New measure** and add these one at a time:

```DAX
Attempts = COUNTROWS(attempts)

Completed Attempts =
CALCULATE(COUNTROWS(attempts), attempts[completed] = TRUE())

Completion Rate = DIVIDE([Completed Attempts], [Attempts], 0)

Total Kills = SUM(attempts[kills])

Deaths = CALCULATE(COUNTROWS(attempts), attempts[dead] = TRUE())

Average Survival Seconds = AVERAGE(attempts[wall_seconds])

Confirmed Pickups =
CALCULATE(COUNTROWS(pickups), pickups[confirmed] = TRUE())

Judge Failures = COUNTROWS(judge_findings)

Unique Sectors Visited = DISTINCTCOUNT(steps[current_sector])
```

## 5. Build the first report page

Create a page called **Campaign Overview**:

- Cards: Attempts, Completion Rate, Total Kills, Deaths and Confirmed Pickups.
- Clustered column chart: `run_id` on the X-axis and kills/items as values.
- Line chart: step on the X-axis and health on the Y-axis, with run_id as legend.
- Matrix: run_id as rows, finding_type as columns, count of findings as values.
- Slicers: map, run_id, completed and judge.

Create a second page called **Navigation and Decisions**:

- Table: step, mode, objective, actions, health and current_sector.
- Sankey custom visual if permitted, using from_sector → to_sector and count.
  A normal matrix is a safe alternative when custom visuals are unavailable.
- Scatter chart: player_x and player_y, colored by mode. Filter to one run at a
  time because lines between unrelated runs are misleading.

## 6. Refresh after another attempt

Run the export command again to overwrite the CSVs, then select **Home →
Refresh** in Power BI. Keep the same column names in the export script so the
report does not break.

## How to edit this yourself

- Edit explanatory text in `README.md` or this file with VS Code.
- The export columns are defined near the top of
  `examples/doom/export_power_bi.py`. Add a column to `STEP_FIELDS` only when
  that field is present in the trace.
- If you add a new table column, refresh Power BI and check its data type in
  Power Query.
- Add one change at a time, run the exporter, and inspect the CSV before editing
  visuals. This makes mistakes easy to isolate.
- Keep claims evidence-based: a kill count is not mission completion, and an
  offline simulator result is not proof of DoomSol website performance.
