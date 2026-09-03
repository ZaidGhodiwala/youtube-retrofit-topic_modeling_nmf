# YouTube Retrofit Topic Modelling with NMF

A reproducible Natural Language Processing and unsupervised machine-learning pipeline for identifying latent knowledge-sharing structures in YouTube discussions about domestic energy retrofit.

Developed as part of paid Research Associate work at Nottingham Trent University, the project uses **TF-IDF and Non-negative Matrix Factorisation (NMF)** to examine what people discuss and how knowledge is exchanged across a large corpus of retrofit-related YouTube comments.

## Project overview

- **42,487** validated substantive comments
- **1,159** unique retrofit-related YouTube videos in the audited corpus
- **42,443** comments and **1,158** videos in the final modelling matrix
- **14,427** TF-IDF features
- **1,013,411** non-zero sparse-matrix entries
- Candidate NMF models evaluated at **k = 8, 10, 12, 14, 16 and 20**
- Repeated stability testing using **10 independent 80% corpus subsamples**
- Parallel preprocessing-sensitivity analysis
- Repository interpretation based on an **inclusive k = 16 NMF solution**
- Reproducibility controls using **SHA-256 verification, deterministic configuration, model serialisation, sparse-matrix persistence and audit logging**

**Core stack:** Python 3.11, pandas, NumPy, SciPy, scikit-learn, joblib, TF-IDF, NMF and sparse matrices.

## Research objective

The project investigates latent structures that emerge directly from commenter language without forcing the unsupervised model to reproduce categories already defined elsewhere in the wider research.

Questions include:

- What discussion structures emerge independently from the comment corpus?
- Do technical retrofit subjects form identifiable latent components?
- Do knowledge-sharing behaviours emerge separately from technical subject matter?
- Does practical troubleshooting appear organically?
- Do clarification, recommendation-seeking and prospective decision-making form distinguishable structures?
- How sensitive are the discovered topics to preprocessing choices?
- How stable are topics when the underlying corpus is repeatedly perturbed?
- Are technical topics more concentrated within particular videos than cross-cutting knowledge-sharing behaviours?

The wider methodological aim is to compare researcher-defined structures with data-driven latent structures while treating neither as ground truth.

## Why NMF?

NMF provides an additive decomposition of the TF-IDF matrix into:

- a **document-topic matrix**, representing topic strength within each comment
- a **topic-term matrix**, representing vocabulary associated with each latent component

The NMF workflow was developed separately from the existing rule-based and LLM-assisted qualitative analysis. Model selection, validation and interpretation were performed independently to reduce the risk of forcing latent topics to mirror predefined categories.

## Corpus audit

The source pipeline progressed through the following denominators:

| Stage | Comments | Videos |
|---|---:|---:|
| Extracted raw comments | 80,153 | - |
| Cleaned comments | 80,142 | - |
| Validated substantive corpus | 42,487 | 1,159 |
| Final primary NMF matrix | 42,443 | 1,158 |

The validated substantive corpus contained:

- 42,487 unique comment IDs
- zero missing modelling texts
- zero blank modelling texts
- zero duplicate comment IDs
- minimum comment length of three words

A total of 44 comments became zero vectors after conservative TF-IDF vocabulary filtering. These exclusions were explicitly recorded rather than silently removed. One video was represented only by a comment that became a zero vector, producing the final modelling denominator of 1,158 videos.

Repeated text was retained where comment IDs were unique because repeated wording may represent authentic user behaviour rather than extraction duplication.

## Conservative text preprocessing

The preprocessing pipeline was designed for short, technically oriented social-media text.

It includes:

- Unicode NFKC normalisation
- HTML entity decoding
- apostrophe standardisation
- URL removal
- lowercasing
- controlled contraction expansion
- whitespace normalisation

Aggressive stemming and lemmatisation were deliberately avoided to preserve domain-specific terminology and maintain interpretable topic-term outputs.

Negation and interrogative terms such as `not`, `no`, `never`, `without`, `how`, `why`, `what`, `should`, `could` and `would` were retained because they can carry meaningful information about troubleshooting, uncertainty, disagreement, help-seeking and decision-making.

## TF-IDF feature engineering

The primary representation uses:

- unigrams and bigrams
- minimum document frequency: 10
- maximum document frequency: 85%
- sublinear term-frequency scaling
- smoothed inverse-document-frequency weighting
- L2 normalisation
- Unicode-aware tokenisation
- technical-token preservation

### Primary representation

| Metric | Value |
|---|---:|
| Modelled comments | 42,443 |
| TF-IDF features | 14,427 |
| Unigrams | 6,433 |
| Bigrams | 7,994 |
| Non-zero entries | 1,013,411 |
| Matrix density | ~0.001655 |

A structured vocabulary audit confirmed that **31 of 32 predefined technical anchor concepts** survived preprocessing, including terms relating to heat pumps, spray foam, vapour barriers, solar panels, wall insulation, heat loss, underfloor heating, cavity walls and airflow.

## Preprocessing-sensitivity design

The project does not assume that one preprocessing configuration is objectively correct.

Two parallel representations are constructed:

### Inclusive primary representation

Retains technical language alongside potentially meaningful social and evaluative language, including expressions of:

- gratitude
- praise
- recommendations
- perceived helpfulness
- practical outcomes
- credibility and endorsement

### Content-focused sensitivity representation

Removes a deliberately narrow set of platform and strong-praise terms such as `video`, `YouTube`, `subscribe`, `watching`, `thanks`, `great`, `excellent` and `awesome`.

Potentially meaningful words such as `worked`, `helpful`, `fixed`, `solved`, `failed` and `recommend` remain.

The content-focused representation contains:

- 42,424 comments
- 14,084 features
- 982,601 non-zero matrix entries
- 31 of 32 retained technical anchor concepts

This provides a direct robustness test of whether major latent structures survive a targeted change in social and platform vocabulary.

## Candidate NMF modelling

Candidate models were fitted at:

```text
k = 8, 10, 12, 14, 16, 20
```

Configuration:

- NNDSVDa initialisation
- coordinate-descent optimisation
- Frobenius reconstruction loss
- fixed random state
- maximum 500 iterations
- convergence tolerance of `1e-4`
- no L1 topic regularisation
- no L2 topic regularisation

The pipeline persists fitted models, document-topic matrices, vectorisers, topic-term matrices, evaluation metrics, representative comments and audit artefacts.

## Multi-criterion model evaluation

Model selection was deliberately not based on a single metric.

Candidate models were assessed using:

- **reconstruction error**
- **NPMI topic coherence**
- **topic diversity**
- **inter-topic cosine similarity**
- **topic-size diagnostics**
- **representative-comment review**

The strongest models after quantitative screening were `k = 12`, `k = 16` and `k = 20`.

Following structured interpretability review, `k = 12` and `k = 16` were advanced to repeated stability testing.

## Repeated stability testing

Candidate topics were tested under repeated perturbation of the corpus using:

- 10 independent repetitions
- 80% corpus subsamples
- identical NMF configuration
- Hungarian topic matching
- topic-vector cosine similarity
- top-term Jaccard similarity
- pairwise run stability
- topic-prevalence stability

Corpus subsampling was used because NNDSVDa is substantially deterministic on an unchanged matrix. Perturbing the underlying data therefore provides a more informative robustness test than repeatedly refitting the same matrix.

### Stability results

| Metric | k = 12 | k = 16 |
|---|---:|---:|
| Mean matched-topic cosine similarity | 0.918 | **0.947** |
| Median matched-topic cosine similarity | 0.995 | **0.998** |
| Mean top-20-term Jaccard similarity | 0.708 | **0.784** |
| Mean pairwise run similarity | 0.880 | **0.917** |
| Mean absolute prevalence difference | 1.28 pp | **0.89 pp** |
| Convergence-warning subsample runs | 2 | **0** |

The 16-topic solution performed more strongly across the principal global stability measures.

## Preprocessing-sensitivity results

The 16-topic dimensionality was refitted using the content-focused representation and matched back to the inclusive representation using Hungarian matching over shared vocabulary.

| Metric | Result |
|---|---:|
| Mean matched-topic cosine similarity | 0.893 |
| Median matched-topic cosine similarity | 0.994 |
| Topics matched at cosine >= 0.90 | 81.2% |
| Topics matched at cosine >= 0.80 | 87.5% |
| Mean top-term Jaccard similarity | 0.696 |
| Mean absolute prevalence difference | 1.13 pp |
| Dominant-topic agreement | 72.1% |
| Adjusted Rand Index | 0.518 |

The major technical topics remained highly stable.

The principal sensitivity occurred in the component representing praise, gratitude and social endorsement. This component weakened when strong-praise vocabulary was deliberately removed, which is substantively consistent with the purpose of the sensitivity test.

## Locked 16-topic interpretation

The repository contains a SHA-256-locked interpretation of the 16-topic solution so that downstream analysis cannot silently alter topic labels.

| Category | Topic | Interpretation |
|---|---:|---|
| Mixed/contextual | 1 | Situated household retrofit experience and building context |
| Technical | 2 | Spray foam and foam insulation |
| Knowledge sharing | 3 | Clarification and specification questions |
| Knowledge sharing | 4 | Implementation, cost and how-to questions |
| Technical | 5 | Heat pumps and heating systems |
| Social interaction | 6 | Praise, gratitude and social endorsement |
| Knowledge sharing | 7 | Alternative methods, challenges and "why" questions |
| Technical | 8 | Solar PV, batteries and power systems |
| Knowledge sharing | 9 | Performance checking and troubleshooting |
| Knowledge sharing | 10 | Implementation follow-up, outcomes and retrospective questions |
| Knowledge sharing | 11 | Product sourcing, access and location questions |
| Technical | 12 | Roof and attic ventilation and airflow |
| Technical | 13 | Wall, cavity and floor insulation |
| Technical | 14 | Thermostat wiring and smart heating controls |
| Technical | 15 | Vapour barriers and moisture control |
| Knowledge sharing | 16 | Prospective choices, recommendations and alternatives |

The final structure contains:

- 7 technical-subject topics
- 7 knowledge-sharing-behaviour topics
- 1 social-interaction topic
- 1 mixed/contextual topic

A central result is that the unsupervised model did not simply divide comments by retrofit technology. It independently recovered both **what people discuss** and **how people exchange knowledge**.

## Soft topic membership

NMF is treated as a continuous multi-topic representation rather than assuming each comment belongs exclusively to one topic.

Across 42,443 modelled comments:

- median dominant relative topic weight: **0.473**
- median secondary relative topic weight: **0.223**
- median dominant-secondary margin: **0.201**
- median effective topic count: **3.81**

This indicates that comments frequently contain meaningful signal from multiple latent components. Dominant-topic assignment is therefore used for descriptive summaries rather than treated as ground truth.

## Video-level structure

Comment-level topic weights are aggregated to the video level across 1,158 videos.

At video level:

- median modelled comments per video: **10**
- interquartile range: **3 to 39**
- median dominant topic weight: **0.275**
- median secondary topic weight: **0.161**
- median dominance margin: **0.090**
- median effective topic count: **8.81**

This shows that full comment sections are substantially more heterogeneous than individual comments.

Technical topics are generally more concentrated within specialised videos, while several knowledge-sharing behaviours are more diffuse across the platform.

For example, the top 10 videos account for:

| Topic | Share of total topic weight in top 10 videos |
|---|---:|
| Spray foam and foam insulation | 40.72% |
| Solar PV, batteries and power systems | 24.04% |
| Thermostat wiring and smart heating controls | 23.28% |
| Heat pumps and heating systems | 22.90% |
| Clarification and specification questions | 8.49% |
| Implementation, cost and how-to questions | 8.85% |
| Performance checking and troubleshooting | 8.88% |

This supports the interpretation that technical subjects can cluster around specialised content, while forms of knowledge exchange operate across many retrofit contexts.

## Repository structure

```text
.
├── analysis/
│   ├── YouTube_Retrofit_NMF_Analysis_Report_Conference_Source.docx
│   └── figures/
├── code/
│   ├── 01_audit_nmf_corpus.py
│   ├── 02_prepare_tfidf_corpus.py
│   ├── 03_review_tfidf_vocabulary.py
│   ├── 04_freeze_parallel_tfidf_representations.py
│   ├── 05_fit_candidate_nmf_models.py
│   ├── 06_prepare_candidate_interpretability_review.py
│   ├── 06b_run_nmf_stability_tests.py
│   ├── 07_run_preprocessing_sensitivity.py
│   ├── 07b_prepare_final_k16_topic_review.py
│   ├── 07c_freeze_final_k16_interpretation.py
│   ├── 08a_analyse_locked_nmf_solution.py
│   ├── 08b_analyse_nmf_video_context.py
│   ├── 08c_analyse_nmf_engagement.py
│   └── 08d_equal_comment_sampling_robustness.py
├── config/
├── data/
│   ├── input/
│   └── processed/
├── outputs/
│   ├── audit/
│   ├── models/
│   ├── review/
│   └── tables/
├── requirements_initial.txt
└── .gitattributes
```

## Pipeline

```text
01  Audit source corpus
 ↓
02  Prepare conservative TF-IDF corpus
 ↓
03  Review vocabulary and technical-term retention
 ↓
04  Freeze parallel TF-IDF representations
 ↓
05  Fit and evaluate candidate NMF models
 ↓
06  Prepare candidate interpretability review
 ↓
06b Run repeated NMF stability tests
 ↓
07  Run preprocessing-sensitivity analysis
 ↓
07b Prepare final k=16 topic review
 ↓
07c Lock interpretation with SHA-256
 ↓
08a Analyse comment-level topic structure
 ↓
08b Analyse video-level topic context
 ↓
08c Analyse engagement relationships
 ↓
08d Test robustness to unequal comment volume
```

## Reproducibility

The workflow includes:

- deterministic modelling configuration
- fixed random states
- SHA-256 source and artefact verification
- explicit expected row and video counts
- zero-vector auditing
- saved TF-IDF vectorisers
- persisted sparse matrices
- serialised NMF models
- stored document-topic and topic-term matrices
- structured configuration files
- stage-specific audit reports
- saved candidate diagnostics
- preprocessing-sensitivity testing
- repeated subsampling stability testing
- cryptographically locked topic interpretation

These controls are intended to make modelling choices traceable and reduce silent analytical drift.

## Installation

The workflow was developed using **Python 3.11**.

```bash
git clone https://github.com/ZaidGhodiwala/youtube-retrofit-topic_modeling_nmf.git
cd youtube-retrofit-topic_modeling_nmf

python -m venv .venv
```

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements_initial.txt
```

## Running the core workflow

Run the scripts from the repository root in numerical order:

```bash
python code/01_audit_nmf_corpus.py
python code/02_prepare_tfidf_corpus.py
python code/03_review_tfidf_vocabulary.py
python code/04_freeze_parallel_tfidf_representations.py
python code/05_fit_candidate_nmf_models.py
python code/06_prepare_candidate_interpretability_review.py
python code/06b_run_nmf_stability_tests.py
python code/07_run_preprocessing_sensitivity.py
python code/07b_prepare_final_k16_topic_review.py
python code/07c_freeze_final_k16_interpretation.py
python code/08a_analyse_locked_nmf_solution.py
python code/08b_analyse_nmf_video_context.py
```

Stages `08c` and `08d` reference an additional project-level metadata file named `metadata_engagement_analysis/integrated_video_analysis_master.csv`, which is not included in this repository. Generated downstream tables and audit outputs from those analyses are included.

Some saved configuration files preserve original machine-specific paths as part of the analytical record. Core modelling scripts derive their main paths relative to the repository root.

## Technologies and methods

**Programming and data:** Python, pandas, NumPy, SciPy, scikit-learn, joblib, sparse matrices

**NLP and machine learning:** TF-IDF, NMF, unsupervised learning, topic modelling, text normalisation

**Evaluation:** NPMI coherence, reconstruction error, topic diversity, cosine similarity, Jaccard similarity, Adjusted Rand Index

**Robustness:** repeated corpus subsampling, Hungarian topic matching, preprocessing-sensitivity analysis, equal-comment resampling

**Reproducibility:** SHA-256 verification, model serialisation, persistent matrices, deterministic configuration, audit logging

## Research context

This project forms part of a wider research stream examining how YouTube functions as an informal environment for sharing domestic energy-retrofit knowledge.

The unsupervised analysis is intended as an independent analytical layer alongside rule-based and qualitative approaches. It does not treat machine-generated topics as definitive categories. Instead, it provides an additional empirical view of the corpus that can be compared with researcher-defined structures through methodological triangulation.

## Author

**Zaid Ghodiwala**  
BSc (Hons) Computer Science, Nottingham Trent University  
Research Associate, Nottingham Trent University

[LinkedIn](https://www.linkedin.com/in/zaid-ghodiwala/) | [GitHub](https://github.com/ZaidGhodiwala)
