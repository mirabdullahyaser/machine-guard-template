# Data

| File | What it is |
|---|---|
| `raw/ai4i2020_original.csv` | The file exactly as published by UCI. Do not edit. |
| `raw/ai4i2020.csv` | The same data with course column names. Made by `python scripts/download_data.py`. |

Later days add more batches here (a messy batch for data validation, a "next quarter" batch for drift).

## Columns

| Column | Meaning |
|---|---|
| `udi` | Row id. Not a feature. |
| `product_id` | Product serial number. Not a feature. |
| `type` | Product quality variant: L (low, 60%), M (medium, 30%), H (high, 10%) |
| `air_temp_k` | Air temperature, kelvin |
| `process_temp_k` | Process temperature, kelvin |
| `rotational_speed_rpm` | Spindle speed, revolutions per minute |
| `torque_nm` | Torque, newton-metres |
| `tool_wear_min` | Minutes the current tool has been in use |
| `machine_failure` | **Target.** 1 if the machine failed on this run |
| `twf`, `hdf`, `pwf`, `osf`, `rnf` | Failure modes: tool wear, heat dissipation, power, overstrain, random. These are recorded *after* the failure, so they are **never** used as model inputs. |

## Licence and credit

Stephan Matzka, "AI4I 2020 Predictive Maintenance Dataset", UCI Machine Learning Repository, 2020.
https://doi.org/10.24432/C5HS5C. Licensed under CC BY 4.0, which allows sharing and adapting with credit.
