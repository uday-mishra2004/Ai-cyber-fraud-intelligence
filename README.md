# AI Cyber-Fraud & Digital Trust Intelligence Platform

A modular Python research platform for transaction fraud classification, unusual-transaction review and possible SMS phishing detection, with MySQL analytics and an interactive Streamlit dashboard.

**Status:** real transaction and SMS datasets are now loaded into local MySQL: 284,807 transactions and 5,971 messages. Live SQL analytics are available; model training/evaluation remains pending. No public deployment is claimed. See [local MySQL operation](docs/local_mysql.md) and [verified load metadata](reports/mysql_data_load.json).

## Implemented capabilities

| Area | Functionality |
| --- | --- |
| Python / NumPy / Pandas | Validated inputs, descriptive statistics and reproducible processing |
| EDA / statistics | Class imbalance, amount/text distributions, duplicates, outliers and descriptive correlations |
| MySQL / SQLAlchemy | Atomic checksum-keyed ingestion, provenance and 18 bound analytics queries |
| Transaction fraud | Majority baseline, weighted Logistic Regression and Random Forest; chronological holdouts |
| Anomaly review | Isolation Forest and robust statistical baselines; reference-only fitting and label-free calibration |
| NLP | TF-IDF + Logistic Regression with duplicate/template-grouped SMS holdouts |
| Optional neural model | Compact TF-IDF/SVD neural experiment using the verified baseline holdout |
| Visualization | Matplotlib/Seaborn reports and interactive Plotly charts |
| Streamlit | Seven views: overview, transaction analytics/risk, anomaly review, messages, performance and methodology |
| Engineering | Pytest, mocked database checks, optional isolated MySQL integration and artifact compatibility guards |

An anomaly is not necessarily fraud. Text scores do not prove that a message is fraudulent or safe. The datasets have no established identity join; there is no overall customer trust score.

## Quick start

Tested on Windows with **Python 3.12.14**. From the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://127.0.0.1:8501` and stop with Ctrl+C. The server binds to localhost. Navigation works without credentials/models; unavailable views explain missing sources and never simulate results.

`requirements-lock.txt` records the verified environment, including optional neural dependencies. It is an exact-version snapshot, not a hash-locked or cross-platform guarantee. For maintained ranges use `requirements.txt`, or `requirements-neural.txt` for the full suite/neural experiment. Preserve training environments or retrain when changing unsupported dependency versions.

A [fresh-environment installation and startup check](reports/execution_readiness.md) verified all 62 pins, reran the full suite (156 passed, 2 optional MySQL tests skipped), and received HTTP 200 from the localhost application. That earlier snapshot preceded dataset acquisition. Real data and local MySQL are now available; trained project artifacts remain absent. This is not a hosted-deployment claim.

## Configure sources and MySQL

1. Read [dataset sources/licences](docs/datasets.md), acquire original files and inspect actual fields/labels.
2. Place local data at `data/raw/creditcard.csv` and `data/raw/phishing.csv`; do not commit either.
3. Install MySQL **>=8.0.16** in strict mode. Follow [setup](docs/database.md) and [schema documentation](docs/schema.md).
4. Copy the template only if no local configuration exists:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
# Fill credentials locally, then check:
.\.venv\Scripts\python.exe -m database.cli check
# Initialize with a schema-admin account:
.\.venv\Scripts\python.exe -m database.cli init
```

Use a separate SELECT-only account for the dashboard and suitable privileges for imports. Environment variables override `.env`. Never put credentials into commands, source files or screenshots. MySQL powers analytics; model training/inference uses files and can run independently.

See [reproducible training/inference commands](docs/reproducibility.md). Existing run directories are not overwritten; successful runs save fitted preprocessing, metadata, thresholds and split manifests.

## Evaluation and verification

Latest completed quality run, **2026-10-12**: **162 passed, 0 skipped, 0 failures**, with one existing Seaborn deprecation warning. This latest run includes three live MySQL integration tests against a separately configured isolated test database; normal test runs keep them opt-in. Dependency and compilation checks passed. These are software checks on artificial fixtures, not performance benchmarks.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

| Objective | Measured real-data results |
| --- | --- |
| Supervised transaction fraud | Not available |
| Anomaly fraud capture at review budgets | Not available |
| SMS TF-IDF baseline | Not available |
| Neural versus baseline comparison | Not available |

No best model, production accuracy, prevented-loss KPI or neural improvement is claimed. Read the [methodology/results guide](docs/methodology.md), [consolidated evaluation](reports/model_evaluation_report.md), [quality review](reports/integration_quality_review.md) and [test instructions](docs/testing.md).

## Repository layout

```text
app.py                  Streamlit entry point
eda.py                  EDA and reusable statistical plots
dashboard/              Views, validated services and Plotly factories
database/  sql/         Persistence, schema and analytics queries
fraud/  anomaly/        Transaction classification and anomaly pipelines
phishing/  neural_text/ SMS baseline and optional neural experiment
tests/                  Unit/component and opt-in MySQL tests
docs/  reports/         Documentation and evidence-based reports
data/  artifacts/       Local sources/run outputs, excluded from Git
```

## Documentation

- [Final portfolio and interview package](docs/interview_package.md): resume bullets, presentation scripts, 25 Q&As, 10 SQL exercises and verification checklist
- [Architecture](docs/architecture.md)
- [Local execution verification and deployment boundaries](docs/execution_and_deployment.md)
- [Dataset sources and licences](docs/datasets.md)
- [MySQL setup](docs/database.md) and [schema/query patterns](docs/schema.md)
- [Training and inference](docs/reproducibility.md)
- [Methodology and results](docs/methodology.md)
- [Dashboard usage](docs/dashboard.md) and [screenshot instructions](docs/screenshots.md)
- [Limitations and responsible use](docs/responsible_use.md)
- [Verified features and future work](docs/feature_status.md)
- [Testing and publication checks](docs/testing.md)

A [real local MySQL transaction-analytics screenshot](docs/screenshots/mysql-transaction-analytics.png) was captured on 2026-10-12 after the verified import. It shows aggregate class imbalance, not model performance or a hosted deployment. Follow the screenshot instructions for additional privacy-reviewed views.

## Licence and attribution

Datasets and dependencies retain their own licences. Mendeley lists CC BY 4.0; Kaggle's public metadata identifies the transaction database/content licences (see the dataset guide). No raw data is redistributed. **A project source-code licence has not yet been selected by the owner.** GitHub visibility alone is not an open-source licence grant.
