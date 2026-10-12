"""Reproducible descriptive EDA. Run `python eda.py --help` for input options."""
from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent


def save_plot(fig, path):
    """Save a figure and always close it to bound memory use."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_counts(series, title, path):
    counts = series.fillna("<missing>").astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="steelblue")
    for i, count in enumerate(counts):
        ax.text(i, count, f"{count:,}\n{count / len(series):.1%}", ha="center", va="bottom")
    ax.set(title=title, xlabel="Label", ylabel="Records")
    ax.margins(y=.25)
    save_plot(fig, path)


def plot_distribution(frame, value, label, title, path):
    """Plot all observations without silently trimming distribution tails."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=frame, x=value, hue=label, bins=40, stat="density",
                 common_norm=False, element="step", ax=axes[0])
    sns.ecdfplot(data=frame, x=value, hue=label, ax=axes[1])
    axes[0].set_title(title)
    axes[1].set_title("Empirical cumulative distribution")
    save_plot(fig, path)


def plot_heatmap(corr, path):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap="vlag", ax=ax)
    ax.set_title("Spearman feature correlations (target excluded)")
    save_plot(fig, path)


def normalize_message(text):
    """Conservative duplicate key; preserve numbers, punctuation, and URLs."""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def template_message(text):
    """Heuristic campaign key; collisions require review, not automatic deletion."""
    text = normalize_message(text)
    text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]+\b", " <email> ", text)
    return re.sub(r"\d+", "<number>", text)


def markdown_table(frame):
    """Markdown without an extra tabulate dependency."""
    frame = frame.reset_index()
    rows = [[str(c) for c in frame.columns]] + [
        [f"{v:.5g}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(rows[0]) + " |",
                       "| " + " | ".join(["---"] * len(rows[0])) + " |"] +
                      ["| " + " | ".join(row) + " |" for row in rows[1:]])


def numeric_summary(frame):
    stats = frame.describe(percentiles=[.01, .25, .5, .75, .95, .99]).T
    stats["missing"] = frame.isna().sum()
    stats["skew"] = frame.skew()
    return stats


def provenance(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"Input: `{Path(path).name}`; SHA-256: `{digest.hexdigest()}`."


def analyse_transactions(frame, output):
    required = {"Class", "Amount", "Time"} | {f"V{i}" for i in range(1, 29)}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing transaction columns: {sorted(missing)}")
    if frame.empty or frame["Class"].isna().any() or not frame["Class"].isin([0, 1]).all():
        raise ValueError("Transactions need nonempty data and binary, nonmissing Class.")
    features = frame[sorted(required - {"Class"})].apply(pd.to_numeric, errors="raise")
    if np.isinf(features.to_numpy(dtype=float)).any():
        raise ValueError("Transaction features contain infinity.")
    if (features["Amount"].dropna() < 0).any():
        raise ValueError("Negative Amount: inspect the source before log transformation.")
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    stats = numeric_summary(features)
    stats.to_csv(tables / "transaction_descriptive_statistics.csv")
    amounts = frame.groupby("Class")["Amount"].describe(percentiles=[.5, .95, .99])
    amounts.to_csv(tables / "transaction_amount_by_class.csv")
    corr = features.corr(method="spearman")
    corr.to_csv(tables / "transaction_spearman.csv")
    q1, q3 = features.quantile(.25), features.quantile(.75)
    iqr = q3 - q1
    flagged = ((features < q1 - 1.5 * iqr) | (features > q3 + 1.5 * iqr))
    outliers = pd.DataFrame({"observed": features.count(), "iqr_flags": flagged.sum(),
                             "iqr_flag_rate": flagged.sum() / features.count(), "iqr": iqr})
    outliers.to_csv(tables / "transaction_outlier_flags.csv")
    plot_counts(frame["Class"], "Transaction labels", figures / "transaction_labels.png")
    plot_distribution(frame, "Amount", "Class", "Transaction amounts", figures / "transaction_amount.png")
    logged = frame.assign(log1p_amount=np.log1p(frame["Amount"]))
    plot_distribution(logged, "log1p_amount", "Class", "log(1 + amount)", figures / "transaction_log_amount.png")
    plot_heatmap(corr, figures / "transaction_correlations.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    outliers["iqr_flag_rate"].sort_values().plot.barh(ax=ax)
    ax.set(title="Potential outliers: 1.5 × IQR rule", xlabel="Flagged fraction of observed values")
    save_plot(fig, figures / "transaction_outliers.png")
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    strongest = pairs.loc[pairs.abs().nlargest(5).index]
    return (f"## Transaction findings\n\nRecords: {len(frame):,}; fraud labels: {int(frame['Class'].sum()):,} "
            f"({frame['Class'].mean():.3%}). Exact duplicate rows beyond first occurrence: {frame.duplicated().sum():,}.\n\n"
            "### Amount statistics by class\n\n" + markdown_table(amounts) +
            "\n\n### Largest absolute feature correlations\n\n" + markdown_table(strongest.rename("Spearman rho")) +
            f"\n\nMissing feature values: {features.isna().sum().sum():,}. "
            "Correlations use pairwise complete observations; inspect missing counts before comparing pairs. "
            "These are descriptive associations, with no significance or causal claim. "
            "IQR flags identify candidates for review, not fraud or invalid observations. "
            "A zero IQR can flag every noncentral observation; no records are removed. "
            "Time is elapsed seconds; calendar dates and time of day cannot be inferred.\n")


def analyse_messages(frame, text_column, label_column, output):
    if text_column not in frame or label_column not in frame or frame.empty:
        raise ValueError("Messages require nonempty data and the specified text/label columns.")
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    text = frame[text_column].fillna("").astype(str)
    labels = frame[label_column].fillna("<missing>").astype(str)
    normal = text.map(normalize_message)
    template = text.map(template_message)
    data = pd.DataFrame({"label": labels, "characters": text.str.len(),
                         "words": text.str.split().str.len()})
    lengths = data.groupby("label")[["characters", "words"]].describe()
    lengths.to_csv(tables / "message_length_statistics.csv")
    groups = pd.DataFrame({"key": normal, "label": labels}).groupby("key")["label"].agg(["size", "nunique"])
    templates = pd.DataFrame({"key": template, "label": labels}).groupby("key")["label"].agg(["size", "nunique"])
    # Do not export raw texts or duplicate keys: they may contain personal information.
    duplicate_audit = pd.DataFrame({"group_hash": [hashlib.sha256(k.encode()).hexdigest() for k in groups.index],
                                     "size": groups["size"].values, "label_count": groups["nunique"].values})
    duplicate_audit.to_csv(tables / "message_duplicate_groups.csv", index=False)
    artefacts = pd.DataFrame({"label": labels,
        "url_present": text.str.contains(r"https?://|www\.", case=False, regex=True),
        "email_present": text.str.contains(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]+\b", regex=True),
        "digit_present": text.str.contains(r"\d", regex=True),
        "replacement_character": text.str.contains("\ufffd", regex=False),
        "html_marker": text.str.contains(r"<[^>]+>", regex=True),
        "label_word_present": text.str.contains(r"\b(?:ham|spam|smishing|phishing)\b", case=False, regex=True)})
    rates = artefacts.groupby("label").mean()
    rates.to_csv(tables / "message_artefact_rates.csv")
    plot_counts(labels, "Message source labels", figures / "message_labels.png")
    for value in ["characters", "words"]:
        plot_distribution(data, value, "label", f"Message length: {value}", figures / f"message_{value}.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(rates, annot=True, fmt=".2f", vmin=0, vmax=1, ax=ax)
    ax.set_title("Artefact presence by source label (descriptive)")
    save_plot(fig, figures / "message_artefacts.png")
    term_rows = []
    for label in sorted(labels.unique()):
        # Document frequency, on unique normalized messages, avoids repeat campaigns dominating.
        documents = normal[labels.eq(label)].drop_duplicates()
        tokens = documents.map(lambda s: set(re.findall(r"\b[a-z]{2,}\b", s)))
        counts = pd.Series([term for row in tokens for term in row], dtype="str").value_counts().head(15)
        for term, count in counts.items():
            term_rows.append({"label": label, "term": term, "document_count": int(count),
                              "document_fraction": count / len(documents)})
        if not counts.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            counts.sort_values().plot.barh(ax=ax)
            ax.set(title=f"Common terms: label {label}", xlabel="Unique messages containing term")
            save_plot(fig, figures / f"message_terms_{hashlib.sha256(label.encode()).hexdigest()[:8]}.png")
    pd.DataFrame(term_rows).to_csv(tables / "message_common_terms.csv", index=False)
    missing_text = frame[text_column].isna().sum()
    return (f"## Phishing/message findings\n\nRecords: {len(frame):,}; missing texts: {missing_text:,}; "
            f"empty/whitespace texts: {normal.eq('').sum():,}; missing labels: {frame[label_column].isna().sum():,}.\n\n" +
            markdown_table(labels.value_counts().rename("records")) +
            f"\n\nExact text duplicates beyond first: {text.duplicated().sum():,}; "
            f"normalized duplicates beyond first: {normal.duplicated().sum():,}; "
            f"normalized groups with conflicting labels: {groups['nunique'].gt(1).sum():,}. "
            f"Heuristic template groups containing multiple messages: {templates['size'].gt(1).sum():,}.\n\n"
            "Keep normalized duplicates together across splits. Review conflicting labels before training. "
            "Template matches are candidates for campaign grouping, not confirmed duplicates; this heuristic does not "
            "detect every semantic paraphrase. Missing/empty messages must be resolved before modelling.\n\n"
            "Artefact rates and common-term plots are screening evidence, not proof of leakage. "
            "Review label words, formatting, encoding replacement characters and source-specific headers manually. "
            "URL presence can be a legitimate signal as well as a source shortcut. Compare performance with and without "
            "suspect features on grouped validation data. Existing non-text columns are not used as model inputs here. "
            "Common terms use unique normalized documents, English alphabetic tokens, and no stop-word removal; "
            "they are not TF-IDF weights or statistically significant predictors. Raw messages are not exported. "
            "Term plots may still expose identifiers and require review before publication.\n")


ASSUMPTIONS = """## Statistical assumptions and limitations

- This report is descriptive. No hypothesis tests or p-values are produced: no prespecified confirmatory hypothesis or defensible independence structure is available. Therefore no relationship is claimed statistically significant.
- Means and standard deviations can be dominated by heavy tails; compare medians, quantiles, ECDFs and log-transformed amounts. Log plots use log(1 + amount), retain zeros, and do not establish normality.
- Spearman correlation describes monotonic association, not causation. PCA feature meanings are unavailable; correlations cannot establish fraud mechanisms. Multiple inspected pairs increase the risk of selecting chance patterns.
- Transactions may be temporally dependent and messages may share campaigns. Ordinary independent-sample tests or confidence intervals would require justification; future inference should consider temporal/block or campaign-level resampling and multiple-comparison correction.
- IQR flags are distribution-dependent heuristics. Rare legitimate transactions can be outliers, while fraud can appear ordinary.
- Source labels are preserved. Spam is not automatically phishing; verify the source label mapping before binary classification. Label distribution reflects this sample, not deployment prevalence.
- Run modelling EDA on training data after freezing grouped/temporal splits. A full-corpus audit is suitable for schema and duplicate checks, but feature selection using held-out labels contaminates evaluation.
- Dataset age, collection sources, OCR and annotation quality limit generalization. No causal, production-readiness or model-performance claims follow from EDA.

## Outputs

Figures are saved under `reports/figures`; numerical tables under `reports/tables`.
No raw message text is included in this report. Counts are measured from supplied files only.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=Path, default=ROOT / "data/raw/creditcard.csv")
    parser.add_argument("--messages", type=Path, default=ROOT / "data/raw/phishing.csv")
    parser.add_argument("--text-column", default="TEXT")
    parser.add_argument("--label-column", default="LABEL")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    parser.add_argument("--scope", choices=["audit", "training"], default="audit")
    args = parser.parse_args()
    sns.set_theme(style="whitegrid")
    args.output.mkdir(parents=True, exist_ok=True)
    parts = [f"# Exploratory data analysis\n\nScope: **{args.scope}**. "
             "Audit mode does not certify that input records belong to the training split.\n"]
    for name, path in [("Transaction", args.transactions), ("Phishing/message", args.messages)]:
        if not path.is_file():
            parts.append(f"## {name} findings\n\nPending: input `{path.name}` is absent. "
                         "No statistics or figures were generated for this dataset.\n")
            continue
        frame = pd.read_csv(path)
        result = (analyse_transactions(frame, args.output) if name == "Transaction" else
                  analyse_messages(frame, args.text_column, args.label_column, args.output))
        parts.extend([result, provenance(path)])
    parts.append(ASSUMPTIONS)
    summary = args.output / "eda_summary.md"
    summary.write_text("\n\n".join(parts), encoding="utf-8")
    print(f"Report saved: {summary}")


if __name__ == "__main__":
    main()
