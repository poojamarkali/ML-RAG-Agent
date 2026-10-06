"""Database tables + sample data. SQLite locally, Postgres in docker -- only
DATABASE_URL changes. To add your own data: define a Table here and fill it."""

from pathlib import Path

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, create_engine, func, insert, select

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
]

SEED_EXPERIMENTS = [
    ("EXP-101", "Random Forest", '{"n_estimators": 100, "max_depth": 5, "min_samples_split": 2}', 0.965, 0.972, "2026-03-10"),
    ("EXP-102", "Random Forest", '{"n_estimators": 200, "max_depth": 8, "min_samples_split": 4}', 0.978, 0.981, "2026-03-12"),
    ("EXP-103", "Kernel SVM", '{"C": 10.0, "kernel": "rbf", "gamma": "scale"}', 0.971, 0.975, "2026-03-15"),
    ("EXP-104", "Gradient Boosting", '{"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3}', 0.982, 0.987, "2026-03-18"),
]


def init_db():
    """Create the tables and add the sample rows (only if the tables are empty)."""
    if engine.url.get_backend_name() == "sqlite" and engine.url.database:
        Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
    metadata.create_all(engine)

    with engine.begin() as conn:
        for table, rows in [(model_benchmarks, SEED_BENCHMARKS), (hyperparameter_experiments, SEED_EXPERIMENTS)]:
            if conn.execute(select(func.count()).select_from(table)).scalar() == 0:
                cols = [c.name for c in table.columns][1:]  # skip id
                conn.execute(insert(table), [dict(zip(cols, row)) for row in rows])
