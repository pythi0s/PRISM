"""Build a v4-styled dissertation report from saved PRISM experiment artifacts."""

import argparse
import json
import math
import statistics
import tempfile
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


SEEDS = (42, 123, 2026)
NAVY = "17365D"
PALE = "F3F6FA"
INK = "243447"
MUTED = "526575"
BLUE = "315C82"
RUST = "B65F3C"
GREEN = "46836B"


def load_runs(artifact_root):
    runs = {}
    for seed in SEEDS:
        path = artifact_root / f"seed_{seed}"
        state = json.loads((path / "run_state.json").read_text(encoding="utf-8"))
        if state.get("last_recorded_status") != "complete" or set(state.get("completed", [])) != {
                "generate", "features", "train-alignment", "train-risk", "evaluate"}:
            raise ValueError(f"Seed {seed} is not a complete evaluated run")
        runs[seed] = {
            "path": path,
            "state": state,
            "metrics": json.loads((path / "metrics.json").read_text(encoding="utf-8")),
            "features": json.loads((path / "features.json").read_text(encoding="utf-8")),
            "dataset": json.loads((path / "dataset.json").read_text(encoding="utf-8")),
            "history": json.loads((path / "finetuning.json").read_text(encoding="utf-8")),
            "alignment_model": joblib.load(path / "alignment.joblib"),
            "risk_model": joblib.load(path / "risk.joblib"),
        }
    return runs


def mean_sd(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    if not values:
        return None, None
    return statistics.mean(values), statistics.pstdev(values)


def fmt(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def fmt_ms(values, digits=3):
    mean, deviation = mean_sd(values)
    return "N/A" if mean is None else f"{mean:.{digits}f} +/- {deviation:.{digits}f}"


def style_document(document):
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)
    borders = OxmlElement("w:pgBorders")
    borders.set(qn("w:offsetFrom"), "page")
    for edge in ("top", "left", "bottom", "right"):
        border = OxmlElement(f"w:{edge}")
        for key, value in (("val", "single"), ("sz", "4"), ("space", "24"), ("color", "auto")):
            border.set(qn(f"w:{key}"), value)
        borders.append(border)
    section._sectPr.append(borders)

    settings = document.settings.element
    update_fields = OxmlElement("w:updateFields")
    update_fields.set(qn("w:val"), "true")
    settings.append(update_fields)

    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.12
    for name, size in (("Title", 22), ("Heading 1", 24), ("Heading 2", 20), ("Heading 3", 20)):
        style = document.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(NAVY)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(14 if name != "Title" else 0)
        style.paragraph_format.space_after = Pt(7)

    header = section.header.paragraphs[0]
    header.text = "BITS PILANI WILP | PRISM DISSERTATION RESULTS"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs:
        run.font.name = "Calibri"
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string(MUTED)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("PRISM | November 2026 | Page ")
    run.font.name = "Calibri"
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string(MUTED)
    page_field = OxmlElement("w:fldSimple")
    page_field.set(qn("w:instr"), "PAGE")
    page_run = OxmlElement("w:r")
    page_text = OxmlElement("w:t")
    page_text.text = "1"
    page_run.append(page_text)
    page_field.append(page_run)
    footer._p.append(page_field)
    between = footer.add_run(" of ")
    between.font.name = "Calibri"
    between.font.size = Pt(8)
    between.font.color.rgb = RGBColor.from_string(MUTED)
    total_field = OxmlElement("w:fldSimple")
    total_field.set(qn("w:instr"), "NUMPAGES")
    total_run = OxmlElement("w:r")
    total_text = OxmlElement("w:t")
    total_text.text = "1"
    total_run.append(total_text)
    total_field.append(total_run)
    footer._p.append(total_field)


def shade(cell, fill):
    properties = cell._tc.get_or_add_tcPr()
    shd = properties.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        properties.append(shd)
    shd.set(qn("w:fill"), fill)


def repeat_table_header(row):
    properties = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    properties.append(repeat)


def set_cell_text(cell, text, header=False, size=8.5):
    cell.text = ""
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.space_before = Pt(2)
    run = paragraph.add_run(str(text))
    run.font.name = "Calibri"
    run.font.size = Pt(size)
    run.font.bold = header
    run.font.color.rgb = RGBColor.from_string("FFFFFF" if header else INK)
    if header:
        shade(cell, NAVY)


def add_table(document, headers, rows, widths=None, font_size=8.5):
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    repeat_table_header(table.rows[0])
    for i, heading in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], heading, header=True, size=font_size)
    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for column, value in enumerate(values):
            set_cell_text(cells[column], value, size=font_size)
            if row_index % 2 == 0:
                shade(cells[column], PALE)
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)
    document.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_body(document, text, italic=False):
    paragraph = document.add_paragraph(style="Normal")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = paragraph.add_run(text)
    run.italic = italic
    return paragraph


def add_bullets(document, items):
    for item in items:
        paragraph = document.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.add_run(item)


def add_caption(document, text):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    run.italic = True
    run.font.name = "Calibri"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor.from_string(MUTED)


def make_figures(runs, folder):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelcolor": INK,
                         "text.color": INK, "axes.edgecolor": "#AAB5BF", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})
    seeds = list(SEEDS)
    paths = {}

    figure, axis = plt.subplots(figsize=(7.1, 3.8))
    x = list(range(len(seeds)))
    width = 0.34
    risk = [runs[s]["metrics"]["overall"]["risk"]["average_precision"] for s in seeds]
    misalignment = [runs[s]["metrics"]["overall"]["alignment_misaligned_positive"]["average_precision"] for s in seeds]
    axis.bar([v - width / 2 for v in x], risk, width, color=f"#{RUST}", label="Risk AP")
    axis.bar([v + width / 2 for v in x], misalignment, width, color=f"#{BLUE}", label="Misalignment AP")
    axis.set_xticks(x, [str(seed) for seed in seeds])
    axis.set_xlabel("Random seed")
    axis.set_ylabel("Average precision")
    axis.set_ylim(0, 1.05)
    axis.legend(frameon=False, ncol=2, loc="lower right")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    paths["seed_metrics"] = folder / "seed_metrics.png"
    figure.savefig(paths["seed_metrics"], dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.1, 3.8))
    difficulties = ("easy", "medium", "hard")
    width = 0.34
    risk = [statistics.mean(runs[s]["metrics"]["difficulty"][d]["risk"]["average_precision"]
                            for s in seeds) for d in difficulties]
    misalignment = [statistics.mean(runs[s]["metrics"]["difficulty"][d]["alignment_misaligned_positive"]["average_precision"]
                                    for s in seeds) for d in difficulties]
    positions = list(range(len(difficulties)))
    axis.bar([v - width / 2 for v in positions], risk, width, color=f"#{RUST}", label="Risk AP")
    axis.bar([v + width / 2 for v in positions], misalignment, width, color=f"#{BLUE}", label="Misalignment AP")
    axis.set_xticks(positions, [d.title() for d in difficulties])
    axis.set_ylabel("Average precision")
    axis.set_ylim(0, 1.05)
    axis.legend(frameon=False, ncol=2, loc="lower right")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    paths["difficulty"] = folder / "difficulty_metrics.png"
    figure.savefig(paths["difficulty"], dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.1, 4.1))
    labels = ["Lexical", "Frozen BGE\n+ lexical", "BGE semantic", "Fine-tuned BGE\n+ lexical",
              "Frozen BGE\n+ cross-encoder", "Fine-tuned BGE\n+ cross-encoder", "MiniLM fixed 0.7"]
    keys = ["lexical", "encoder_lexical", "encoder_semantic", "finetuned_lexical",
            "encoder_hybrid", "finetuned_hybrid", "legacy_minilm"]
    values = []
    deviations = []
    for key in keys:
        scores = []
        for seed in seeds:
            entry = runs[seed]["metrics"]["alignment_ablations"][key]
            item = entry.get("calibrated_test", entry)
            scores.append(item["average_precision"])
        mean, sd = mean_sd(scores)
        values.append(mean)
        deviations.append(sd)
    colors = ["#95A3AE", f"#{BLUE}", "#6E8FAA", f"#{RUST}", f"#{GREEN}", "#BD805F", "#C9D0D6"]
    positions = list(range(len(keys)))
    axis.bar(positions, values, yerr=deviations, color=colors, capsize=3, error_kw={"elinewidth": 1})
    axis.set_xticks(positions, labels, rotation=0, fontsize=8)
    axis.set_ylabel("Misalignment average precision")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    paths["ablations"] = folder / "alignment_ablations.png"
    figure.savefig(paths["ablations"], dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.1, 3.9))
    risk_keys = ["dummy", "logistic", "gradient_boosting", "random_forest"]
    labels = ["Dummy", "Logistic", "Gradient\nboosting", "Random\nforest"]
    values = []
    deviations = []
    for key in risk_keys:
        scores = [runs[seed]["metrics"]["risk_baselines_uncalibrated"][key]["average_precision"]
                  for seed in seeds]
        mean, sd = mean_sd(scores)
        values.append(mean)
        deviations.append(sd)
    axis.bar(range(len(risk_keys)), values, yerr=deviations,
             color=["#C9D0D6", f"#{BLUE}", f"#{RUST}", f"#{GREEN}"], capsize=3)
    axis.set_xticks(range(len(risk_keys)), labels)
    axis.set_ylabel("Risk average precision")
    axis.set_ylim(0, 0.75)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    paths["risk_baselines"] = folder / "risk_baselines.png"
    figure.savefig(paths["risk_baselines"], dpi=180)
    plt.close(figure)
    return paths


def add_cover(document):
    p = document.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(40)
    p.add_run("Semantic Alignment and Risk Prediction in Software Pull Requests")

    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("PRISM: Pull Request Intelligence for Semantic Mapping")
    r.bold = True
    r.font.size = Pt(20)
    r.font.color.rgb = RGBColor.from_string(NAVY)

    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Literature Review, Experimental Results, and Limitations")
    r.italic = True
    r.font.size = Pt(14)

    for line in (
        "Dissertation Results Report",
        "Course No.: S1-26_SSWTZG628T | Course Title: Dissertation",
        "Student: Bhagyashri Kiran Bhat | BITS ID: 2022wc15160",
        "MTech - Integrated Software Systems (WASE)",
        "Research Area: Machine Learning",
        "Project carried out at: Wipro Technologies, Pune",
    ):
        p = document.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(7)
        p.add_run(line)

    reference = Path(__file__).parents[1] / "PRISM_BITS_Project_Outline_v4.docx"
    if reference.is_file():
        with ZipFile(reference) as archive:
            logo = BytesIO(archive.read("word/media/image1.jpeg"))
        p = document.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(8)
        p.add_run().add_picture(logo, width=Inches(1.075), height=Inches(1.034))

    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(20)
    r = p.add_run("BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE, PILANI\nWork Integrated Learning Programmes (WILP) Division\nNovember 2026")
    r.bold = True
    r.font.size = Pt(13)
    document.add_page_break()


def add_contents(document):
    document.add_heading("Contents", level=1)
    sections = [
        "Executive Summary", "1. Introduction and Research Questions", "2. Literature Review",
        "3. Research Objectives", "4. Methodology", "5. Results", "6. Discussion",
        "7. Limitations and Threats to Validity", "8. Conclusions and Future Work",
        "References", "Appendix A. Reproducibility and Model Artifacts",
    ]
    for section in sections:
        p = document.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        p.add_run(section)
    document.add_page_break()


def add_references(document):
    references = [
        "Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of NAACL-HLT 2019, 4171-4186. https://doi.org/10.18653/v1/N19-1423",
        "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. Proceedings of EMNLP-IJCNLP 2019, 3982-3992. https://doi.org/10.18653/v1/D19-1410",
        "Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). MTEB: Massive Text Embedding Benchmark. Proceedings of EACL 2023, 2014-2037. https://doi.org/10.18653/v1/2023.eacl-main.148",
        "Xiao, S., Liu, Z., Zhang, P., Muennighoff, N., Lian, D., & Nie, J.-Y. (2024). C-Pack: Packed Resources for General Chinese Embeddings. Proceedings of SIGIR 2024; arXiv:2309.07597. https://arxiv.org/abs/2309.07597",
        "Lin, J., Liu, Y., Zeng, Q., Jiang, M., & Cleland-Huang, J. (2021). Traceability Transformed: Generating More Accurate Links with Pre-Trained BERT Models. Proceedings of ICSE 2021, 324-335. https://doi.org/10.1109/ICSE43902.2021.00040",
        "Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT. arXiv:1901.04085. https://arxiv.org/abs/1901.04085",
        "Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4), 333-389. https://doi.org/10.1561/1500000019",
        "Wei, J., & Zou, K. (2019). EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks. Proceedings of EMNLP-IJCNLP 2019, 6381-6387. https://doi.org/10.18653/v1/D19-1670",
        "Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32. https://doi.org/10.1023/A:1010933404324",
        "Kamei, Y., Shihab, E., Adams, B., Hassan, A. E., Mockus, A., Sinha, A., & Ubayashi, N. (2013). A Large-Scale Empirical Study of Just-in-Time Quality Assurance. IEEE Transactions on Software Engineering, 39(6), 757-773. https://doi.org/10.1109/TSE.2012.70",
        "Niculescu-Mizil, A., & Caruana, R. (2005). Predicting Good Probabilities with Supervised Learning. Proceedings of ICML 2005, 625-632. https://doi.org/10.1145/1102351.1102430",
        "Saito, T., & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLoS ONE, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432",
        "Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems 30. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html",
        "Kapoor, S., & Narayanan, A. (2023). Leakage and the Reproducibility Crisis in Machine-Learning-Based Science. Patterns, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804",
        "BAAI. BAAI/bge-base-en-v1.5 model card and FlagEmbedding resources. Model snapshot: a5beb1e3e68b9ab74eb54cfd186867f64f240e1a. https://huggingface.co/BAAI/bge-base-en-v1.5",
        "Sentence Transformers. MultipleNegativesRankingLoss API documentation. https://sbert.net/docs/package_reference/losses.html",
    ]
    document.add_heading("References", level=1)
    for index, reference in enumerate(references, 1):
        p = document.add_paragraph(style="Normal")
        p.paragraph_format.left_indent = Inches(0.28)
        p.paragraph_format.first_line_indent = Inches(-0.28)
        p.paragraph_format.space_after = Pt(5)
        p.add_run(f"[{index}] {reference}")


def generate_report(artifact_root, output):
    runs = load_runs(artifact_root)
    summary = json.loads((artifact_root / "seed_summary.json").read_text(encoding="utf-8"))
    document = Document()
    style_document(document)
    add_cover(document)
    add_contents(document)

    document.add_heading("Executive Summary", level=1)
    add_body(document,
             "This report presents a literature-grounded evaluation of PRISM, an encoder-based prototype for measuring Jira-issue/pull-request (PR) alignment and estimating review risk. The approved scope deliberately uses a controlled synthetic benchmark and excludes production deployment, organizational data, executable-code analysis, and generative large language models. The completed experiment generated 9,000 PRs per seed for three seeds (27,000 total), used a chronological, task-group-disjoint split, fine-tuned BGE once per seed, and evaluated each frozen pipeline on 1,800 held-out PRs.")
    add_body(document,
             f"Across seeds, misalignment average precision (AP) was {summary['alignment_misaligned_positive']['ap_mean']:.3f} (SD {summary['alignment_misaligned_positive']['ap_std']:.3f}); high-risk AP was {summary['risk']['ap_mean']:.3f} (SD {summary['risk']['ap_std']:.3f}). The alignment task is substantially easier than risk prediction on this generator. Importantly, the final selection chose a frozen-BGE cross-encoder hybrid for seed 42 and frozen-BGE lexical features for seeds 123 and 2026; the fine-tuned variants did not win the selected AP comparisons. Risk selection chose logistic regression for seeds 42 and 123 and Random Forest for seed 2026. These are synthetic benchmark results, not estimates of real-project performance.")
    add_body(document,
             "All seeds completed on CPU using one thread, batch size one, 128-token inputs, one fine-tuning epoch, and no CUDA device. The combined run took approximately 3 hours 55 minutes and peaked at about 3.69 GiB process RSS with no swap. This document reads saved artifacts only; it performs no training.")
    add_table(document, ["Outcome", "Mean across seeds", "Across-seed SD", "Interpretation"], [
        ["Misalignment AP", fmt(summary["alignment_misaligned_positive"]["ap_mean"]), fmt(summary["alignment_misaligned_positive"]["ap_std"]), "High separation on controlled synthetic pairs"],
        ["Risk AP", fmt(summary["risk"]["ap_mean"]), fmt(summary["risk"]["ap_std"]), "Moderate signal above the prevalence baseline"],
        ["Held-out records", "1,800 per seed", "5,400 total", "20% chronological test split per seed"],
    ], widths=[1.25, 1.1, 1.0, 3.25])

    document.add_heading("1. Introduction and Research Questions", level=1)
    add_body(document,
             "Issue trackers record the requested change, whereas pull requests record the proposed implementation. A link between a Jira issue and a PR provides traceability but does not establish that the PR addresses the ticket. Reviewers also need a prioritization signal for changes likely to merit closer inspection. PRISM studies these two tasks separately: semantic alignment classification/retrieval, and synthetic change-risk prediction.")
    add_body(document,
             "The research questions follow the approved outline: (RQ1) can multi-field encoder, lexical, and structural signals distinguish aligned from misaligned synthetic Jira–PR pairs? (RQ2) does cross-encoder rescoring or training-only hard-negative fine-tuning improve held-out alignment over frozen encoders? (RQ3) how well do observable synthetic change features predict hidden, noisy risk labels compared with simple baselines? (RQ4) how do these results vary by benchmark difficulty and misalignment type, and what errors remain?")
    add_body(document,
             "The benchmark is intentionally synthetic. Accordingly, the inferential target is behavior under the generator's assumptions—not real defect occurrence, production reviewer utility, or transfer to a company repository.")

    document.add_heading("2. Literature Review", level=1)
    document.add_heading("2.1 Software traceability and change-risk prediction", level=2)
    add_body(document,
             "Requirements-to-code traceability recovery has long been treated as an information-retrieval problem: a candidate link is ranked by how relevant the implementation artifact appears to the requirement. Lin et al. [5] show that pretrained BERT representations can improve trace-link generation, motivating contextual language models for issue-to-code relationships. The task is not identical to Jira-to-PR alignment: a PR aggregates intent, discussion, and change summaries rather than a single source artifact. That difference motivates evaluating multiple PR fields and a linked-pair classifier rather than assuming a source-code traceability result transfers directly.")
    add_body(document,
             "Just-in-time quality-assurance research studies change-level risk signals such as change size, files/modules, and developer history. Kamei et al. [10] provide large-scale empirical evidence that change metadata can support defect-proneness prioritization in real repositories. PRISM borrows the change-level framing, but its labels and features are synthetic and its latent risk mechanism is designed for this benchmark. Therefore, its results test pipeline behavior under controlled proxy relationships; they do not reproduce the empirical claim of JIT defect prediction.")

    document.add_heading("2.2 Bi-encoders, cross-encoders, and ranking", level=2)
    add_body(document,
             "BERT representations are contextual and useful for language understanding [1], but a standard pairwise BERT scorer must jointly process both members of every pair. SBERT uses Siamese/triplet structures to encode each text independently and compare embeddings with cosine similarity [2]. The separate encoding permits reusable embeddings and efficient retrieval over candidate collections, which suits issue-to-PR candidate ranking. Its trade-off is that independently encoded vectors may miss fine-grained token interactions available when both texts are read together.")
    add_body(document,
             "Cross-encoders address that trade-off by jointly scoring a query/candidate pair. Passage reranking research demonstrates their value for relevance ranking [6], while their pairwise compute cost makes them less suitable for exhaustively scoring every candidate in a large collection. PRISM therefore uses a bi-encoder for candidate retrieval and cross-encoder scores as an additional linked-pair feature. This is a two-stage design: efficient candidate representation followed by more expressive pair scoring.")
    add_body(document,
             "MTEB was proposed because embedding quality varies by task; across its evaluated tasks no single embedding approach dominated universally [3]. This directly cautions against selecting BGE only from a broad benchmark claim. PRISM's synthetic evaluation provides a task-specific comparison, but because its generator is finite and controlled, it still cannot establish superiority on external repositories. The BAAI C-Pack work describes embedding resources and English models [4]; the exact BGE checkpoint and immutable revision used here are recorded in the artifacts rather than inferred from the model-family name.")

    document.add_heading("2.3 Lexical, structural, and synthetic signals", level=2)
    add_body(document,
             "BM25 is a probabilistic ranking function that combines term frequency, document frequency, and length normalization [7]. TF–IDF cosine and identifier overlap supply complementary lexical signals; module/path and branch-name matches encode repository structure that a dense vector may smooth over. In issue-to-PR links, exact identifiers can be useful evidence but can also create shortcuts. PRISM strips literal issue identifiers from text scoring and separates actual relevant-ticket truth from intentionally incorrect linked-ticket fields in its audit data.")
    add_body(document,
             "Synthetic text augmentation can expose classifiers to controlled variation. Wei and Zou's EDA methods use simple word-level transformations to augment text classification data [8]. PRISM uses authored templates, a finite synonym inventory, and explicit partial/unrelated/vague negatives rather than claiming broad natural-language coverage. The benefit is reproducible ground truth and controllable difficulty; the cost is that models can exploit template, vocabulary, and path artifacts. This benchmark therefore needs a dataset card and should be treated as a method stress test, not a proxy for the distribution of organizational Jira or PR text.")

    document.add_heading("2.4 Risk classification, calibration, and interpretation", level=2)
    add_body(document,
             "Random Forests aggregate decorrelated decision trees and provide a nonlinear tabular baseline [9]. Logistic regression supplies a simpler, calibrated-friendly comparison, while gradient boosting tests another tree-based decision boundary. A risk probability is useful only if its scale is meaningful for prioritization. Niculescu-Mizil and Caruana [11] show why probability calibration should be evaluated separately from ranking accuracy; PRISM fits a sigmoid mapping on a dedicated calibration partition and chooses its operating threshold on another held-out development partition.")
    add_body(document,
             "Average precision summarizes the precision–recall ranking and is more informative than ROC AUC alone when positive classes are imbalanced [12]. It is prevalence-dependent, so the positive fraction and task definition must accompany it. PRISM reports ROC AUC alongside AP, F1, precision/recall, Brier score, and confusion counts. SHAP provides additive per-feature attribution values for individual predictions [13]; here it explains an uncalibrated Random Forest baseline, not the selected calibrated risk model and not causal effects.")
    add_body(document,
             "Leakage can inflate machine-learning estimates when information from held-out data or outcome construction enters training [14]. PRISM uses chronological, task-group-disjoint splits, training-only lexical fitting, separate selection/calibration/threshold subsets, and audit-only latent labels. These controls reduce known leakage paths, but they cannot remove generator/model co-design or guarantee external validity.")

    document.add_heading("2.5 Literature synthesis and research gap", level=2)
    add_body(document,
             "Prior work supports the components but not the complete PRISM claim: SBERT enables reusable semantic vectors; cross-encoders offer richer pairwise scoring; traceability work motivates ranking issue-to-artifact links; BM25 and structural matching provide exact signals; and JIT studies motivate change-level prioritization. The gap addressed here is an integrated, leakage-aware prototype that evaluates synthetic Jira–PR alignment and review-risk signals separately, with explicit ground truth, calibration, retrieval, model comparisons, error breakdowns, and attribution. The gap is methodological and prototype-focused; this study does not claim a new general-purpose embedding architecture or a validated production risk model.")

    document.add_heading("3. Research Objectives and Hypotheses", level=1)
    add_bullets(document, [
        "O1: Construct a seeded synthetic benchmark with paraphrases, synonyms, hard and easy negatives, label/text noise, difficulty labels, and hidden probabilistic risk outcomes.",
        "O2: Compare frozen and fine-tuned BGE representations, lexical/structural signals, and cross-encoder rescoring for linked-pair alignment and ticket retrieval.",
        "O3: Select calibrated alignment classifiers using development data and report thresholds and operating metrics on untouched chronological test data.",
        "O4: Compare class-weighted RF, logistic regression, gradient boosting, and a dummy risk baseline on observable metadata; explain the RF baseline with SHAP.",
        "O5: Report results per seed, difficulty, and misalignment type, including ablations, bootstrap intervals, and false-positive/false-negative analysis.",
    ])
    add_body(document,
             "The directional expectation was that semantic and hybrid alignment signals would exceed lexical-only and description-only MiniLM baselines; that hard-negative fine-tuning could improve the frozen encoder; and that observable proxies could predict noisy hidden risk labels above a dummy baseline. These are evaluated as benchmark-specific hypotheses, not presumed true outcomes.")

    document.add_heading("4. Methodology", level=1)
    document.add_heading("4.1 Benchmark generation and provenance", level=2)
    add_body(document,
             "All tickets, PRs, authors, file paths, metadata, and outcomes were generated locally. No organizational dataset, external repository history, executable source code, generative LLM, or production event was used. The task catalog spans six modules and twenty task/paraphrase pairs per module. Jira and PR titles/descriptions use independently selected templates and seeded substitutions. Linked pairs are aligned, or are intentionally partially related (same module/different task), unrelated, or vague. The audit file retains actual relevant Jira IDs, clean labels, latent factors, and risk probabilities; the predictors do not receive those oracle columns.")
    add_body(document,
             "Each seed generated 3,000 PRs at each of three difficulty levels: 9,000 per seed and 27,000 total. Requested partial-negative shares within the negative branch were 10%, 45%, and 75% for easy/medium/hard; text noise was 2%, 10%, and 25%; label-flip noise was 0%, 2%, and 5%. Because sampling is stochastic, realized counts differ slightly by seed. The risk label is a Bernoulli outcome from a hidden logistic function of latent complexity, author familiarity, defect propensity, and noise; observed change metadata are noisy proxies rather than empirical defect determinants.")
    add_body(document,
             "The test set contains 1,800 PRs per seed (5,400 across the study). Retrieval excludes queries with no relevant ticket and reports their count; the remaining query count varies by seed. Each PR’s linked Jira ID may be intentionally wrong, so ranking uses separate audit truth. Candidate tickets are split-local and filtered by availability at the PR timestamp.")

    document.add_heading("4.2 Split protocol and leakage controls", level=2)
    add_table(document, ["Partition", "Fraction", "PRs per seed", "Use"], [
        ["Train", "60%", "5,400", "Fit encoders/classifiers and lexical transforms"],
        ["Selection", "10%", "900", "Select checkpoints, variants, and hyperparameters"],
        ["Calibration", "5%", "450", "Fit sigmoid probability calibrators"],
        ["Threshold", "5%", "450", "Select operating thresholds"],
        ["Test", "20%", "1,800", "Final evaluation only"],
    ], widths=[1.0, 0.75, 0.9, 3.95])
    add_body(document,
             "The partitions are chronological; canonical task groups remain in one partition across difficulty settings; candidate issues precede PR review times; author activity uses only earlier events. Vocabulary/IDF and model preprocessing are fit on training data. Calibration and decision-threshold sets are separate from model selection. Test metrics did not select the model or threshold. These safeguards address specific leakage paths; they do not eliminate template-family dependence or generator/model co-design.")

    document.add_heading("4.3 Alignment models and features", level=2)
    add_body(document,
             "The bi-encoder was BAAI/bge-base-en-v1.5, with immutable revision recorded in each seed's features.json. Semantic cosine similarities were computed between Jira text and PR title, description, code-change summary, and changed-path text. Lexical/structural features included train-fitted TF–IDF cosine, BM25, token/identifier overlap, module-to-path match, branch overlap, and missing-description indicators. The cross-encoder was cross-encoder/ms-marco-MiniLM-L-6-v2. Its scalar score was treated as a feature, not as a probability.")
    add_body(document,
             "Fine-tuning used Multiple Negatives Ranking Loss with one training-only hard negative per positive batch example, filtering candidate tasks from positive groups. Although each seed has 5,400 training rows, the implementation deduplicates positive anchors by canonical task group and therefore used 72 unique positive groups per epoch. The checkpoint evaluator used the selection split. This limited positive diversity is an important interpretation constraint.")
    add_body(document,
             "Alignment model variants were lexical-only; frozen-BGE semantic/lexical/hybrid; fine-tuned-BGE lexical/hybrid; and description-only MiniLM using the legacy fixed cosine 0.7 threshold. Logistic regression and gradient boosting were selected using the selection set. Calibrated outputs and thresholds were fitted using dedicated partitions. The selected feature set and classifier vary by seed, so the report preserves seed-level selection rather than presenting a single universal winner.")

    document.add_heading("4.4 Risk models and explanations", level=2)
    add_body(document,
             "Risk predictors were observable pre-review change metadata: lines added/deleted, file count, critical-file count, test-file count, comment count, prior author PR count, days since prior PR, and churn ratio. IDs, split/difficulty, audit latent values, clean labels, and alignment outcomes were excluded. The candidate classifiers were a prior-only DummyClassifier, class-weighted logistic regression, sklearn gradient boosting, and a class-weighted Random Forest. Three-fold StratifiedGroupKFold cross-validation ran within the training partition. Candidate selection used development average precision; calibration and threshold selection used their own subsets. SHAP explanations were computed for the separate uncalibrated RF baseline, with additivity checks; they are associations, not causes.")

    document.add_heading("4.5 Experimental environment and reproducibility", level=2)
    add_body(document,
             "All three full runs used Python 3.10, CPU execution, CUDA hidden, one CPU thread, batch size one, maximum sequence length 128, and one fine-tuning epoch per seed. Native thread pools were recorded at one. Seed-specific model/artifact revisions and run-state files are saved alongside model checkpoints, datasets, split manifests, metrics, predictions, and reports. Total wall-clock time was 3:54:46; peak process RSS was 3,688,608 KiB (about 3.52 GiB); no swap was used. The GPU remained idle. The reproducible artifact bundle is archived at full_bge_cpu.tar.gz with a SHA-256 checksum.")

    document.add_heading("4.6 Evaluation measures", level=2)
    add_body(document,
             "For binary classification, the report uses ROC AUC, average precision (AP; the project's PR-AUC summary), F1, precision, recall, and Brier score. Misaligned-positive and aligned-positive metrics are separated. Risk metrics treat high risk as the positive class. MRR and Recall@1/3/5 evaluate the bi-encoder ranking over available candidates; vague/no-relevant queries are excluded and counted. Group-bootstrap intervals resample task groups within the test partition. Across-seed SD is descriptive variability over the three generator/model seeds; it is not a confidence interval over real projects.")

    document.add_heading("5. Results", level=1)
    document.add_heading("5.1 Primary held-out performance", level=2)
    risk_summary = summary["risk"]
    alignment_summary = summary["alignment_misaligned_positive"]
    add_table(document, ["Task / positive class", "ROC AUC", "AP", "F1", "Precision", "Recall", "Brier"], [
        ["Risk / high-risk", *[fmt_ms([runs[s]["metrics"]["overall"]["risk"][key] for s in SEEDS])
                                for key in ("roc_auc", "average_precision", "f1", "precision", "recall", "brier")]],
        ["Alignment / misaligned", *[fmt_ms([runs[s]["metrics"]["overall"]["alignment_misaligned_positive"][key] for s in SEEDS])
                                     for key in ("roc_auc", "average_precision", "f1", "precision", "recall", "brier")]],
        ["Alignment / aligned", *[fmt_ms([runs[s]["metrics"]["overall"]["alignment_aligned_positive"][key] for s in SEEDS])
                                   for key in ("roc_auc", "average_precision", "f1", "precision", "recall", "brier")]],
    ], widths=[1.2, 0.82, 0.82, 0.72, 0.9, 0.75, 0.82], font_size=7.6)
    add_caption(document, "Table 1. Held-out mean +/- population SD across seeds 42, 123, and 2026.")
    add_body(document,
             f"Misalignment ranking was strong on this benchmark: AP {alignment_summary['ap_mean']:.3f} (SD {alignment_summary['ap_std']:.3f}), with mean precision {mean_sd([runs[s]['metrics']['overall']['alignment_misaligned_positive']['precision'] for s in SEEDS])[0]:.3f} and recall {mean_sd([runs[s]['metrics']['overall']['alignment_misaligned_positive']['recall'] for s in SEEDS])[0]:.3f}. Risk prediction was materially weaker: AP {risk_summary['ap_mean']:.3f} (SD {risk_summary['ap_std']:.3f}), mean ROC AUC {mean_sd([runs[s]['metrics']['overall']['risk']['roc_auc'] for s in SEEDS])[0]:.3f}, and Brier score {mean_sd([runs[s]['metrics']['overall']['risk']['brier'] for s in SEEDS])[0]:.3f}. Risk AP exceeds each seed's high-risk prevalence, but absolute performance remains moderate and synthetic.")
    with tempfile.TemporaryDirectory(prefix="prism-report-") as temp:
        figures = make_figures(runs, Path(temp))
        document.add_picture(str(figures["seed_metrics"]), width=Inches(6.7))
        add_caption(document, "Figure 1. Held-out risk and misalignment average precision by seed.")

        document.add_heading("5.2 Seed-level model selection and threshold variability", level=2)
        selected_rows = []
        for seed in SEEDS:
            a = runs[seed]["alignment_model"]
            r = runs[seed]["risk_model"]
            selected_rows.append([str(seed), runs[seed]["features"]["selected_feature_set"], a.name,
                                  fmt(a.threshold, 3), r.name, fmt(r.threshold, 3)])
        add_table(document, ["Seed", "Alignment features", "Alignment classifier", "Misalignment threshold", "Risk classifier", "High-risk threshold"],
                  selected_rows, widths=[0.55, 1.3, 1.0, 1.1, 1.0, 1.25], font_size=8)
        add_caption(document, "Table 2. Models were selected on development data; thresholds were selected on the dedicated threshold partition.")
        add_body(document,
                 "The selected alignment feature set differed across seeds: seed 42 selected the frozen-BGE/cross-encoder hybrid with gradient boosting, while seeds 123 and 2026 selected frozen-BGE lexical-plus-semantic features with logistic regression. Risk selection chose logistic regression for seeds 42 and 123 and Random Forest for seed 2026. The chosen misalignment thresholds also varied (approximately 0.223–0.795), reinforcing that the calibrated decision threshold is operating-point and seed dependent rather than a universal constant.")

        document.add_heading("5.3 Difficulty and misalignment-type analysis", level=2)
        difficulty_rows = []
        for difficulty in ("easy", "medium", "hard"):
            metrics = [runs[s]["metrics"]["difficulty"][difficulty] for s in SEEDS]
            difficulty_rows.append([difficulty.title(),
                                    fmt_ms([m["risk"]["average_precision"] for m in metrics]),
                                    fmt_ms([m["risk"]["f1"] for m in metrics]),
                                    fmt_ms([m["alignment_misaligned_positive"]["average_precision"] for m in metrics]),
                                    fmt_ms([m["alignment_misaligned_positive"]["f1"] for m in metrics])])
        add_table(document, ["Difficulty", "Risk AP", "Risk F1", "Misalignment AP", "Misalignment F1"],
                  difficulty_rows, widths=[1.0, 1.25, 1.1, 1.5, 1.45])
        add_caption(document, "Table 3. Per-difficulty held-out metrics averaged over the three seeds.")
        document.add_picture(str(figures["difficulty"]), width=Inches(6.7))
        add_caption(document, "Figure 2. Mean held-out risk and misalignment AP by benchmark difficulty.")
        type_rows = []
        for kind in ("partial", "unrelated", "vague"):
            values = [runs[s]["metrics"]["misalignment_type"][kind]["alignment_misaligned_positive"] for s in SEEDS]
            type_rows.append([kind.title(), ", ".join(str(v["records"]) for v in values),
                              fmt_ms([v["average_precision"] for v in values]),
                              fmt_ms([v["f1"] for v in values])])
        add_table(document, ["Misalignment type", "Test records by seed", "AP mean +/- SD", "F1 mean +/- SD"],
                  type_rows, widths=[1.35, 1.55, 1.65, 1.55])
        add_caption(document, "Table 4. Misaligned-positive metrics for partial, unrelated, and vague negatives. The 'none' stratum is not a meaningful positive-class ranking task.")
        add_body(document,
                 "Hard cases reduce alignment AP and F1 relative to easy cases, although the mean hard-set AP remains high. Partial, unrelated, and vague negatives are separately reported; 'none' is the aligned stratum and its misaligned-positive AP is not interpreted as a hard-negative score. Small strata and label flips can produce single-class or noise-sensitive metrics, so counts accompany scores.")

        document.add_heading("5.4 Ablations and the effect of fine-tuning", level=2)
        ablation_keys = ["lexical", "legacy_minilm", "encoder_semantic", "encoder_lexical",
                         "finetuned_lexical", "encoder_hybrid", "finetuned_hybrid"]
        ablation_labels = {"lexical": "Lexical only", "legacy_minilm": "MiniLM description / fixed 0.7",
                           "encoder_semantic": "Frozen BGE semantic", "encoder_lexical": "Frozen BGE + lexical",
                           "finetuned_lexical": "Fine-tuned BGE + lexical", "encoder_hybrid": "Frozen BGE + cross-encoder",
                           "finetuned_hybrid": "Fine-tuned BGE + cross-encoder"}
        ablation_rows = []
        for key in ablation_keys:
            scores = []
            f1s = []
            for seed in SEEDS:
                entry = runs[seed]["metrics"]["alignment_ablations"][key]
                result = entry.get("calibrated_test", entry)
                scores.append(result.get("average_precision"))
                f1s.append(result.get("f1"))
            ablation_rows.append([ablation_labels[key], fmt_ms(scores), fmt_ms(f1s)])
        add_table(document, ["Alignment variant", "Misalignment AP", "Misalignment F1"],
                  ablation_rows, widths=[3.4, 1.7, 1.5])
        add_caption(document, "Table 5. Held-out ablation performance, mean +/- population SD across seeds.")
        document.add_picture(str(figures["ablations"]), width=Inches(6.7))
        add_caption(document, "Figure 3. Misalignment AP across alignment ablations; error bars show across-seed population SD.")
        add_body(document,
                 "Fine-tuning did not improve held-out AP in these runs: frozen-BGE lexical-plus-semantic AP averaged approximately 0.962 versus approximately 0.947 for fine-tuned BGE plus lexical features; the frozen hybrid averaged approximately 0.963 versus approximately 0.947 for the fine-tuned hybrid. Fine-tuned variants had higher recall in some operating points but could lower precision/F1. The selected models favored frozen BGE for all three seeds. This is a negative result for the stated fine-tuning hypothesis on this generator and one-epoch training setup, not evidence that fine-tuning is universally ineffective.")

        document.add_heading("5.5 Risk baselines, retrieval, and error analysis", level=2)
        risk_rows = []
        for key, label in (("dummy", "Dummy prior"), ("logistic", "Logistic regression"),
                           ("gradient_boosting", "Gradient boosting"), ("random_forest", "Random Forest")):
            entries = [runs[s]["metrics"]["risk_baselines_uncalibrated"][key] for s in SEEDS]
            risk_rows.append([label, fmt_ms([x["average_precision"] for x in entries]),
                              fmt_ms([x["f1"] for x in entries]), fmt_ms([x["brier"] for x in entries])])
        add_table(document, ["Risk baseline (uncalibrated)", "AP", "F1", "Brier"],
                  risk_rows, widths=[2.5, 1.5, 1.4, 1.2])
        add_caption(document, "Table 6. Uncalibrated test-set baselines; the main risk metrics above are for each seed's selected/calibrated model.")
        document.add_picture(str(figures["risk_baselines"]), width=Inches(6.7))
        add_caption(document, "Figure 4. Uncalibrated risk baseline AP by model.")
        retrieval_rows = []
        for seed in SEEDS:
            values = runs[seed]["metrics"]["overall"]["bi_encoder_retrieval"]
            retrieval_rows.append([str(seed), str(values["eligible_queries"]),
                                   str(values["excluded_no_relevant_ticket"]), fmt(values["mrr"]),
                                   fmt(values["recall_at_1"]), fmt(values["recall_at_3"]), fmt(values["recall_at_5"])])
        add_table(document, ["Seed", "Eligible", "Excluded", "MRR", "R@1", "R@3", "R@5"],
                  retrieval_rows, widths=[0.65, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0])
        add_caption(document, "Table 7. Retrieval over split-local, time-available Jira candidates; no-relevant-ticket queries are excluded and counted.")
        total_errors = {task: {
            "fp": sum(runs[s]["metrics"]["error_analysis"][task]["false_positive_count"] for s in SEEDS),
            "fn": sum(runs[s]["metrics"]["error_analysis"][task]["false_negative_count"] for s in SEEDS),
        } for task in ("misalignment", "risk")}
        add_table(document, ["Task", "False positives (all seeds)", "False negatives (all seeds)"], [
            ["Misalignment alert", str(total_errors["misalignment"]["fp"]), str(total_errors["misalignment"]["fn"])],
            ["High-risk alert", str(total_errors["risk"]["fp"]), str(total_errors["risk"]["fn"])],
        ], widths=[2.6, 2.0, 2.0])
        add_caption(document, "Table 8. Error counts at thresholds selected on the dedicated development threshold split; totals combine 5,400 held-out PRs.")
        error_by_difficulty = []
        for difficulty in ("easy", "medium", "hard"):
            for task in ("misalignment", "risk"):
                fp = sum(runs[s]["metrics"]["error_analysis"][task]["false_positives_by_difficulty"].get(difficulty, 0)
                         for s in SEEDS)
                fn = sum(runs[s]["metrics"]["error_analysis"][task]["false_negatives_by_difficulty"].get(difficulty, 0)
                         for s in SEEDS)
                error_by_difficulty.append([difficulty.title(), task.title(), str(fp), str(fn)])
        add_table(document, ["Difficulty", "Task", "False positives", "False negatives"],
                  error_by_difficulty, widths=[1.35, 1.5, 1.7, 1.7])
        add_caption(document, "Table 9. Error counts stratified by difficulty and task.")
        example_rows = []
        for seed in SEEDS:
            analysis = runs[seed]["metrics"]["error_analysis"]
            for task in ("misalignment", "risk"):
                for direction, key in (("FP", "false_positive_examples"), ("FN", "false_negative_examples")):
                    examples = sorted(analysis[task][key], key=lambda row: row["probability"], reverse=(direction == "FP"))
                    for example in examples[:2]:
                        example_rows.append([str(seed), task.title(), direction, example["pr_id"],
                                             example["difficulty"], example["misalignment_type"],
                                             fmt(example["probability"], 3)])
        add_table(document, ["Seed", "Task", "Error", "PR", "Difficulty", "Pair type", "Score"],
                  example_rows, widths=[0.5, 1.1, 0.6, 1.1, 0.9, 1.4, 0.7], font_size=7.6)
        add_caption(document, "Table 10. Two representative false-positive and false-negative examples per seed/task, selected from the saved test error analysis.")
        for seed in SEEDS:
            calibration = runs[seed]["path"] / "calibration.png"
            document.add_picture(str(calibration), width=Inches(5.8))
            add_caption(document, f"Figure 5.{SEEDS.index(seed) + 1}. Saved test-set reliability curves for seed {seed} (risk and alignment).")
        add_body(document,
                 "Retrieval MRR is approximately 0.909 across seeds, with Recall@1 approximately 0.838, Recall@3 approximately 0.985, and Recall@5 approximately 0.998 over eligible queries. Misalignment classification produced 60 false positives and 185 false negatives across the three test sets. The selected risk alerts produced 2,083 false positives and 340 false negatives. The risk operating point is therefore recall-heavy; with this synthetic label distribution, that trade-off would require explicit cost analysis before any operational use. It is not a production recommendation.")

    document.add_heading("6. Discussion", level=1)
    add_body(document,
             "The main positive result is that Jira–PR alignment is learnable within the benchmark's controlled text and candidate construction. Frozen BGE features plus lexical/structural evidence and, in one seed, cross-encoder rescoring produced high misalignment AP across seeds. The sizeable gap between lexical-only and frozen semantic variants is consistent with the task's paraphrase design. However, the finite task catalog, shared template family, module/path construction, and synthetic candidate pools may create cues that a real repository will not share.")
    add_body(document,
             "The direct ablation provides no support for the claim that the implemented one-epoch hard-negative fine-tuning improves the held-out results. Fine-tuning examples collapse many repeated rows onto a limited set of canonical task groups, and the same author-designed catalog shapes both training and evaluation. This limited diversity and the hard-negative selection strategy are plausible explanations, but the experiment does not isolate a causal reason. The report therefore preserves the negative finding rather than selecting a favorable seed or metric.")
    add_body(document,
             "Risk AP is around 0.55, above the roughly 0.33 high-risk prevalence baseline, but the across-seed spread and large number of false positives matter. The risk target is a stochastic label generated from latent assumptions. The result means observable synthetic proxies contain signal about that synthetic target; it does not mean the model predicts real software defects. The weak-to-moderate risk performance is a useful outcome: the hidden probabilistic mechanism prevents exact rule recovery, while still allowing partial predictability.")
    add_body(document,
             "Calibration scores and thresholds vary by seed. AP is threshold-free, while F1/precision/recall depend on a development-selected threshold. These quantities answer different questions and should not be conflated. The model selected on development data is retained per seed; test AP is reported for evaluation only and is not used to select the best seed.")

    document.add_heading("7. Limitations and Threats to Validity", level=1)
    document.add_heading("7.1 Construct validity", level=2)
    add_body(document,
             "'Alignment' is represented by generated task identity and controlled paraphrases; it is not adjudicated by developers or domain experts. 'Risk' is sampled from an assumed latent logistic model, not observed defects, rollbacks, or incidents. Consequently, the labels measure conformance to the generator's specification rather than the intended real-world constructs.")
    document.add_heading("7.2 Internal validity and leakage", level=2)
    add_body(document,
             "Task groups are disjoint across chronological splits; features and transforms exclude hidden audit fields; model selection, calibration, threshold selection, and test evaluation are separated. Nonetheless, the generator and models were designed together, the synthetic template vocabulary is finite, and author activities are simplified. Repeated modules and renderer styles can expose generator fingerprints. These controls reduce direct leakage but cannot remove benchmark-design bias.")
    document.add_heading("7.3 Statistical conclusion validity", level=2)
    add_body(document,
             "Three seeds estimate sensitivity to the implemented generator/model randomness, not uncertainty over independent organizations. A group bootstrap resamples task groups within each held-out split; its interval does not account for a new generator, new repository, or human label disagreement. Per-difficulty subsets have fewer positives, and noisy labels can create rare positives in nominally aligned 'none' cases. AP is sensitive to prevalence. Both counts and metrics must be read together.")
    document.add_heading("7.4 External validity and operational scope", level=2)
    add_body(document,
             "No real Jira or GitHub data, confidential repository, executable code, CI outcome, or production workflow was evaluated. BGE/CrossEncoder revision and CPU protocol are recorded, but a model-card benchmark claim is not evidence of transfer. Real-project evaluation, temporal drift, reviewer acceptance, fairness across authors/teams, and the consequences of false positives/negatives remain open. The risk score is an optional review aid in the prototype, not a defect predictor or automated gate.")

    document.add_heading("8. Conclusions and Future Work", level=1)
    add_body(document,
             "PRISM implements and evaluates a controlled synthetic pipeline combining multi-field BGE semantics, lexical and structural signals, cross-encoder rescoring, calibrated classification, synthetic hidden-risk modeling, classical baselines, retrieval, SHAP, and reviewer-oriented reports. Across three seeds, synthetic misalignment AP was high and stable, whereas synthetic risk AP was moderate. Frozen BGE models were selected in all seeds; one-epoch fine-tuning did not improve test AP in these experiments. These findings answer the benchmark questions but should not be generalized to real repositories.")
    add_bullets(document, [
        "Obtain expert-reviewed real issue/PR pairs and independently adjudicated outcomes before making any external-validity claim.",
        "Expand task, paraphrase, synonym, repository, and author distributions; test under generator shifts and leave-module/repository-out partitions.",
        "Study alternative hard-negative curricula and larger, diverse positive-pair corpora; preserve a locked test set for each design iteration.",
        "Define stakeholder-specific costs for missed misalignment, missed risk, and excess review alerts; select operating points from those costs rather than generic F1 alone.",
        "Audit the final administrative title inconsistency in the source outline before submission: its last page names Predictive CI/CD, while the approved pages 1-9 describe PRISM.",
    ])
    document.add_heading("Appendix A. Reproducibility and Model Artifacts", level=1)
    add_body(document,
             "The complete archive is PRISM/research/artifacts/full_bge_cpu.tar.gz; its checksum is full_bge_cpu.tar.gz.sha256. It contains seed-specific checkpoints, calibrators/classifiers, feature extractors, datasets, split manifests, predictions, metrics, explanations, and HTML reports. Verify with sha256sum -c PRISM/research/artifacts/full_bge_cpu.tar.gz.sha256 and extract into PRISM/research/artifacts. The setup script restores the archive and builds a CPU environment without retraining.")
    artifact_rows = []
    for seed in SEEDS:
        run = runs[seed]
        model = run["alignment_model"]
        risk = run["risk_model"]
        artifact_rows.append([str(seed), run["features"]["selected_feature_set"], model.name,
                              risk.name, str(run["path"]/"bge_finetuned"/"model.safetensors")])
    add_table(document, ["Seed", "Selected features", "Alignment classifier", "Risk classifier", "Fine-tuned encoder weights"],
              artifact_rows, widths=[0.5, 1.25, 1.05, 0.9, 2.9], font_size=7.4)
    add_references(document)

    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate the PRISM dissertation report from saved artifacts; never trains models")
    parser.add_argument("--artifacts", type=Path, default=Path("PRISM/research/artifacts/full_bge_cpu"))
    parser.add_argument("--output", type=Path, default=Path("PRISM/PRISM_Dissertation_Results_Report.docx"))
    args = parser.parse_args(argv)
    output = generate_report(args.artifacts, args.output)
    print(f"Report generated from saved experiment artifacts: {output}")


if __name__ == "__main__":
    main()