# Machine Guard architecture

How the pieces fit together at the end of Day 4. The diagram is ready; fill in the `TODO` lines underneath with your own links and numbers.

```mermaid
---
config:
  theme: base
  fontFamily: "Helvetica, Arial, sans-serif"
  themeVariables:
    primaryColor: "#e8eef6"
    primaryTextColor: "#1f3a5f"
    primaryBorderColor: "#1f3a5f"
    lineColor: "#5b6b7f"
    clusterBkg: "#f8fafc"
    clusterBorder: "#9aa8b8"
    edgeLabelBackground: "#ffffff"
---
flowchart LR
  subgraph BUILD["`Build (codespace)`"]
    direction TB
    DATA[("`**Sensor data**
DVC-versioned CSV`")] --> VAL("`**Validate**
Pandera schema`") --> TRAIN("`**Train**
train.py or Prefect flow`") --> REG[("`**MLflow registry**
@champion, @challenger`")] --> EXP("`**Export**
models/model.joblib`")
  end
  subgraph CI["`Check (CI)`"]
    direction TB
    TEST("`**Tests**
unit + integration`") --> GATE{"`**Quality gate**
recall ≥ 0.75`"} --> IMG("`**Image**
ghcr.io/…/machine-guard`")
  end
  subgraph RUN["`Serve`"]
    direction TB
    API("`**FastAPI**
/health, /predict, API key
Render and kind`") --> LOG[("`**Prediction log**
SQLite + JSON lines`")]
    UI("`**Streamlit page**`") --> API
  end
  subgraph WATCH["`Watch and react`"]
    direction TB
    DRIFT("`**Drift check**
Evidently, weekly`") --> ISSUE("`**GitHub issue**
alert`") --> BOOK("`**Runbook**
retrain or roll back`")
  end
  BUILD ==>|"git push"| CI ==>|"new image"| RUN ==>|"new data"| WATCH
  WATCH -.-> LOOP(["`**Back to Build**
retrain or roll back`"])
  classDef navy fill:#1f3a5f,stroke:#1f3a5f,color:#ffffff,font-weight:bold
  classDef blue fill:#e8eef6,stroke:#1f3a5f,color:#1f3a5f
  classDef amber fill:#fff4d6,stroke:#c98a00,color:#5c3d00
  classDef green fill:#e5f4ea,stroke:#2e7d4f,color:#1d4d31
  classDef red fill:#fdecec,stroke:#b42318,color:#7a1610
  classDef grey fill:#f4f6f8,stroke:#9aa8b8,color:#33414f
  class DATA,VAL,TRAIN,REG,EXP blue
  class TEST,GATE,IMG amber
  class API,LOG,UI green
  class DRIFT,ISSUE,BOOK red
  class LOOP navy
  style BUILD fill:#f8fafc,stroke:#1f3a5f,color:#1f3a5f
  style CI fill:#fffaf0,stroke:#c98a00,color:#5c3d00
  style RUN fill:#f5fbf7,stroke:#2e7d4f,color:#1d4d31
  style WATCH fill:#fff7f7,stroke:#b42318,color:#7a1610
```

## The flow in one paragraph

Sensor data is versioned with DVC and checked by a Pandera schema before anything trains on it. Training logs every run to MLflow and registers the best model as `@champion`. The champion is exported to `models/` and committed, so every environment gets exactly the same model. Each push runs GitHub Actions: tests, the quality gate (recall on failures at least 0.75), then a Docker image pushed to GitHub Container Registry. Render and the kind cluster run that image. Every prediction is logged. Once a week, a drift check compares new data with the training data and opens a GitHub issue if the inputs have moved. The runbook says what to do next: retrain with the Prefect flow, or roll back to the previous model.

## Your system

| Piece | Where | Value |
|---|---|---|
| Repo | GitHub | `TODO: https://github.com/<you>/machine-guard` |
| Live API | Render | `TODO` |
| Streamlit page | Streamlit Community Cloud | `TODO` |
| Image | GHCR | `TODO: ghcr.io/<you>/machine-guard:<tag>` |
| Champion model | MLflow | `TODO: version, recall, precision` |
| CI | GitHub Actions | `TODO: link to the last green run` |
| Drift alert | GitHub Actions + Issues | `TODO: link to the alert issue` |

## Design choices and trade-offs

| Choice | Why | What we gave up |
|---|---|---|
| Model files committed to Git | Docker, CI, Render and kind can all build without the MLflow database | Repo grows by a few MB per model version |
| Recall gate at 0.75 | A missed breakdown costs about 20 false alarms | More false alarms are accepted |
| Weekly drift check, not real-time | Free, simple, enough for slow changes like a new supplier | A sudden sensor fault could go unnoticed for days |
| Promotion moves an alias; shipping is a human step | A person checks before a new model goes live | Slower than fully automatic deployment |
| `TODO: one choice of your own` | | |
