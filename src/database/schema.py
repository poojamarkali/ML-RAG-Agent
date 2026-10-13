"""Database tables + sample data. SQLite locally, Postgres in docker -- only
DATABASE_URL changes. To add your own data: define a Table here and fill it."""

from pathlib import Path

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, create_engine, insert, select

from config import settings

engine = create_engine(settings.database_url)
metadata = MetaData()

model_benchmarks = Table(
    "model_benchmarks", metadata,
    Column("id", Integer, primary_key=True),
    Column("model_name", String, nullable=False),
    Column("category", String, nullable=False),          # classification, clustering, dim_reduction
    Column("dataset", String, nullable=False),
    Column("metric_name", String, nullable=False),       # accuracy, silhouette, ...
    Column("metric_value", Float, nullable=False),
    Column("avg_inference_latency_ms", Float, nullable=False),
    Column("train_time_sec", Float, nullable=False),
    Column("memory_usage_mb", Float, nullable=False),
    Column("is_interpretable", Integer, nullable=False),  # 1 = yes, 0 = black box
)

hyperparameter_experiments = Table(
    "hyperparameter_experiments", metadata,
    Column("id", Integer, primary_key=True),
    Column("experiment_id", String, nullable=False),
    Column("model_name", String, nullable=False),
    Column("hyperparameters", String, nullable=False),   # JSON string
    Column("cross_val_score", Float, nullable=False),
    Column("test_score", Float, nullable=False),
    Column("training_date", String, nullable=False),
)

# Sample numbers for the demo (illustrative, not real measurements). Add rows here and
# run any command that calls init_db() -- rows not yet in the database get inserted.
SEED_BENCHMARKS = [
    ("k-Nearest Neighbors (k=3)", "classification", "breast_cancer", "accuracy", 0.937, 1.25, 0.05, 12.4, 1),
    ("k-Nearest Neighbors (k=7)", "classification", "breast_cancer", "accuracy", 0.944, 1.40, 0.05, 12.5, 1),
    ("Logistic Regression (C=1.0)", "classification", "breast_cancer", "accuracy", 0.958, 0.42, 0.12, 10.1, 1),
    ("Logistic Regression (C=100.0)", "classification", "breast_cancer", "accuracy", 0.965, 0.45, 0.18, 10.2, 1),
    ("Linear SVM (C=1.0)", "classification", "breast_cancer", "accuracy", 0.951, 0.38, 0.15, 11.0, 1),
    ("Kernel SVM (RBF, C=10, gamma=0.1)", "classification", "breast_cancer", "accuracy", 0.972, 1.85, 0.42, 15.3, 0),
    ("Decision Tree (max_depth=4)", "classification", "breast_cancer", "accuracy", 0.951, 0.22, 0.08, 9.8, 1),
    ("Random Forest (n_estimators=100)", "classification", "breast_cancer", "accuracy", 0.979, 4.50, 1.85, 24.6, 0),
    ("Gradient Boosted Trees (lr=0.1)", "classification", "breast_cancer", "accuracy", 0.986, 3.80, 2.40, 21.0, 0),
    ("Multilayer Perceptron (MLP)", "classification", "breast_cancer", "accuracy", 0.965, 2.10, 3.20, 28.5, 0),
    ("Principal Component Analysis (PCA)", "dim_reduction", "breast_cancer", "explained_variance_ratio", 0.925, 0.65, 0.10, 14.0, 1),
    ("t-SNE (perplexity=30)", "dim_reduction", "digits", "kl_divergence", 0.780, 18.50, 12.40, 45.0, 0),
    ("k-Means (k=3)", "clustering", "iris", "silhouette", 0.702, 0.85, 0.08, 11.2, 1),
    ("DBSCAN (eps=0.5, min_samples=5)", "clustering", "breast_cancer", "silhouette", 0.620, 1.15, 0.14, 13.8, 1),
    ("Latent Dirichlet Allocation (LDA)", "clustering", "text_corpus", "perplexity", 42.50, 12.30, 8.50, 65.0, 1),
    ("Gaussian Naive Bayes", "classification", "breast_cancer", "accuracy", 0.937, 0.15, 0.01, 2.4, 1),
    # iris
    ("k-Nearest Neighbors (k=3)", "classification", "iris", "accuracy", 0.974, 0.35, 0.01, 2.1, 1),
    ("Logistic Regression (C=1.0)", "classification", "iris", "accuracy", 0.968, 0.12, 0.03, 1.8, 1),
    ("Decision Tree (max_depth=3)", "classification", "iris", "accuracy", 0.947, 0.08, 0.01, 1.5, 1),
    ("Random Forest (n_estimators=100)", "classification", "iris", "accuracy", 0.961, 2.10, 0.35, 6.2, 0),
    ("Kernel SVM (RBF, C=1, gamma=scale)", "classification", "iris", "accuracy", 0.974, 0.40, 0.02, 2.0, 0),
    ("Agglomerative Clustering (ward, k=3)", "clustering", "iris", "silhouette", 0.554, 0.95, 0.03, 3.1, 1),
    # digits
    ("k-Nearest Neighbors (k=3)", "classification", "digits", "accuracy", 0.987, 4.80, 0.02, 14.3, 1),
    ("Logistic Regression (C=1.0)", "classification", "digits", "accuracy", 0.962, 0.55, 0.95, 9.6, 1),
    ("Random Forest (n_estimators=100)", "classification", "digits", "accuracy", 0.972, 5.20, 1.10, 22.8, 0),
    ("Kernel SVM (RBF, C=10, gamma=0.001)", "classification", "digits", "accuracy", 0.991, 6.10, 0.48, 18.4, 0),
    ("Multilayer Perceptron (MLP)", "classification", "digits", "accuracy", 0.978, 1.90, 6.80, 26.1, 0),
    ("Gaussian Naive Bayes", "classification", "digits", "accuracy", 0.807, 0.30, 0.01, 3.2, 1),
    ("k-Means (k=10)", "clustering", "digits", "adjusted_rand_score", 0.669, 1.20, 0.45, 15.6, 1),
    # regression (california_housing, metric = R^2)
    ("Linear Regression", "regression", "california_housing", "r2_score", 0.606, 0.10, 0.04, 4.8, 1),
    ("Ridge (alpha=1.0)", "regression", "california_housing", "r2_score", 0.606, 0.10, 0.03, 4.8, 1),
    ("Lasso (alpha=0.1)", "regression", "california_housing", "r2_score", 0.548, 0.10, 0.09, 4.8, 1),
    ("Random Forest Regressor (n_estimators=100)", "regression", "california_housing", "r2_score", 0.805, 9.40, 14.20, 180.5, 0),
    ("Gradient Boosted Trees Regressor (lr=0.1)", "regression", "california_housing", "r2_score", 0.776, 3.10, 6.50, 12.3, 0),
    ("k-Nearest Neighbors Regressor (k=5)", "regression", "california_housing", "r2_score", 0.689, 7.80, 0.05, 30.2, 1),
    # faces
    ("Non-Negative Matrix Factorization (NMF, 15 components)", "dim_reduction", "faces", "reconstruction_error", 0.412, 2.40, 9.80, 40.2, 1),
]

SEED_EXPERIMENTS = [
    ("EXP-101", "Random Forest", '{"n_estimators": 100, "max_depth": 5, "min_samples_split": 2}', 0.965, 0.972, "2026-03-10"),
    ("EXP-102", "Random Forest", '{"n_estimators": 200, "max_depth": 8, "min_samples_split": 4}', 0.978, 0.981, "2026-03-12"),
    ("EXP-103", "Kernel SVM", '{"C": 10.0, "kernel": "rbf", "gamma": "scale"}', 0.971, 0.975, "2026-03-15"),
    ("EXP-104", "Gradient Boosting", '{"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3}', 0.982, 0.987, "2026-03-18"),
    ("EXP-105", "Kernel SVM", '{"C": 1.0, "kernel": "rbf", "gamma": 0.01}', 0.962, 0.958, "2026-04-02"),
    ("EXP-106", "Kernel SVM", '{"C": 100.0, "kernel": "rbf", "gamma": 0.001}', 0.976, 0.979, "2026-04-03"),
    ("EXP-107", "Gradient Boosting", '{"n_estimators": 300, "learning_rate": 0.01, "max_depth": 2}', 0.979, 0.983, "2026-04-08"),
    ("EXP-108", "Gradient Boosting", '{"n_estimators": 100, "learning_rate": 0.5, "max_depth": 5}', 0.951, 0.944, "2026-04-09"),
    ("EXP-109", "k-Nearest Neighbors", '{"n_neighbors": 1}', 0.912, 0.902, "2026-04-12"),
    ("EXP-110", "k-Nearest Neighbors", '{"n_neighbors": 9}', 0.935, 0.944, "2026-04-12"),
    ("EXP-111", "Logistic Regression", '{"C": 0.01, "penalty": "l2"}', 0.931, 0.937, "2026-04-15"),
    ("EXP-112", "Logistic Regression", '{"C": 100, "penalty": "l2"}', 0.962, 0.965, "2026-04-15"),
    ("EXP-113", "MLP", '{"hidden_layer_sizes": [100], "alpha": 0.0001}', 0.960, 0.958, "2026-04-20"),
    ("EXP-114", "MLP", '{"hidden_layer_sizes": [10, 10], "alpha": 1.0}', 0.967, 0.972, "2026-04-21"),
]


def insert_missing(conn, table: Table, rows: list[tuple], key: list[str]):
    """Insert the sample rows whose `key` columns aren't in the table yet (safe to run every startup)."""
    cols = [c.name for c in table.columns][1:]  # skip id
    existing = {tuple(r) for r in conn.execute(select(*[table.c[k] for k in key]))}
    new = [dict(zip(cols, row)) for row in rows]
    new = [r for r in new if tuple(r[k] for k in key) not in existing]
    if new:
        conn.execute(insert(table), new)


def init_db():
    """Create the tables and add any sample rows that are missing."""
    if engine.url.get_backend_name() == "sqlite" and engine.url.database:
        Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
    metadata.create_all(engine)

    with engine.begin() as conn:
        insert_missing(conn, model_benchmarks, SEED_BENCHMARKS, ["model_name", "dataset", "metric_name"])
        insert_missing(conn, hyperparameter_experiments, SEED_EXPERIMENTS, ["experiment_id"])
