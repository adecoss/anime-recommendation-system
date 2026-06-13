# Week 10 Recommendation, Ranking, and Evaluation Report

## Task Framing

The Week 10 task is personalized anime recommendation. The supported decision is:

> Given a user's completed liked anime, which titles should be ranked highest as plausible next discoveries?

This is a ranking task rather than exact score prediction. A correct recommendation means that a held-out anime the user liked is ranked near the top of a candidate set.

The intended product is level-aware:

| Level | Product behavior |
| --- | --- |
| Absolute Beginner | guided onboarding, genre/content filters, popularity, high score, short entry points |
| Amateur | similar anime, relation navigation, high-rated anchors, popular/current shows |
| Good | collaborative ranking, current shows, controlled exploration outside comfort genres |
| Pro | long-tail discovery, niche clusters, novelty, graph-aware exploration |

## Data Alignment

The model uses:

- `data/processed/anime_dataset.csv` as the item catalog.
- `data/processed/ratings_processed.csv` as the anonymized interaction layer.
- Positive interactions defined as ratings `>= 7`, matching MAL's own `Good` label.

The candidate pool is restricted to catalog anime with at least 20 positive ratings in the filtered interaction data. The current run aligns 41,928,689 positive interactions to 9,197 candidate anime and evaluates 15,000 users.

## User Controls

The recommender is designed to support user control before ranking:

| Control | Purpose |
| --- | --- |
| Genre filter | include or exclude broad MAL genres |
| Demographic filter | Josei, Kodomo, Seinen, Shoujo, Shounen, 18+ |
| Content rating filter | G, PG, PG-13, R, R+, Rx |
| Explicit toggle | include/exclude Ecchi, Erotica, Hentai, and explicit tags |
| Score floor | keep safer high-confidence titles |
| Episode length filter | limit episode count or total watchtime |
| Year filter | control recency or seasonal discovery |

## Evaluation Protocol

For each eligible user, one liked anime is deterministically held out. The remaining liked anime form the user profile and training data. The held-out user-item pair is removed from the training matrix before evaluation.

Each evaluated ranking contains:

- 1 held-out liked anime
- 100 sampled catalog negatives

Because the rating source has no timestamps, this is not claimed as chronological next-watch prediction. It tests whether the recommender can recover a known liked item from plausible alternatives.

## Systems Compared

| System | Role | Description |
| --- | --- | --- |
| `popularity_baseline` | Baseline | Ranks candidates by global positive interaction count in training data. |
| `latent_svd` | Stronger model | Truncated SVD over the sparse positive user-anime matrix. |
| `hybrid_svd_popularity` | Product-safe blend | Fixed blend of latent SVD score and popularity score. |

Popularity is a serious baseline in anime discovery because discourse often concentrates around current or famous shows. The model only counts as useful if it beats that baseline while still producing explainable recommendations.

## Overall Results

| Method | Hit@10 | NDCG@10 | MRR | Median rank |
| --- | ---: | ---: | ---: | ---: |
| `latent_svd` | 0.910 | 0.716 | 0.658 | 1 |
| `hybrid_svd_popularity` | 0.910 | 0.716 | 0.658 | 1 |
| `popularity_baseline` | 0.770 | 0.501 | 0.430 | 4 |

Both personalized models beat the popularity baseline on every metric. With the `7+` threshold, SVD is the narrow metric winner, while the hybrid has a lower mean rank. The hybrid remains useful as a product-safe option when the interface should avoid overly obscure recommendations for sparse profiles.

## Results by User Level

Offline levels use known liked ratings as a proxy for experience. This is not perfect, but it lets the evaluation ask whether the ranking behaves differently for sparse and dense profiles.

| Level | Hybrid Hit@10 | Hybrid median rank | Evaluated users | Median profile size |
| --- | ---: | ---: | ---: | ---: |
| Absolute Beginner | 0.913 | 1.0 | 4,499 | 25 |
| Amateur | 0.920 | 1.0 | 5,575 | 90 |
| Good | 0.896 | 2.0 | 4,864 | 245 |
| Pro | 0.806 | 3.0 | 62 | 1,153.5 |

The Pro group is tiny in the evaluation sample and should not be overclaimed. It is still useful as a warning: heavy users are harder because obvious titles are often already known, and the recommender should shift toward novelty and niche discovery.

## Beginner Entry-Point Layer

The project also generates `week10_beginner_entrypoint_candidates.csv`. This is not a collaborative model output. It is an onboarding pool for users with no reliable history. It filters out explicit entries and obvious continuation entries, then ranks high-score, high-engagement, shorter or medium-length titles. Examples from the current run include:

- `Sousou no Frieren`
- `Steins;Gate`
- `Koe no Katachi`
- `Kimi no Na wa.`
- `Fullmetal Alchemist: Brotherhood`
- `Shingeki no Kyojin`
- `Vinland Saga`
- `Violet Evergarden`

## Error Analysis

Strong cases occur when the user's known liked anime contain coherent collaborative signals: repeated genres, related communities, franchises, demographic patterns, or shared taste neighborhoods.

Failures occur when:

- the user profile is sparse or broad
- the held-out item reflects a secondary interest
- a user's taste changes over time, which cannot be modeled without timestamps
- a sequel/franchise item should be handled by relation navigation rather than general discovery
- the sampled-negative evaluation is easier than full-catalog recommendation

## Product Interpretation

The final recommender should not mix every signal into one undifferentiated score. The clean product architecture is:

1. Apply user controls: genre, rating, explicit toggle, score, year, and length.
2. Choose the maturity mode: beginner, amateur, good, or pro.
3. Generate candidates from popularity, collaborative SVD, content similarity, and graph relations.
4. Rank candidates with a hybrid score.
5. Use relation edges as contextual navigation, so sequels and side stories appear after a user engages with a main entry rather than as awkward cold recommendations.

## Reproducible Artifacts

Run:

```bash
python src/08_run_week10_recommendation.py
```

Main outputs:

- `artifacts/recommendation/week10_evaluation_metrics.csv`
- `artifacts/recommendation/week10_metrics_by_user_level.csv`
- `artifacts/recommendation/week10_beginner_entrypoint_candidates.csv`
- `artifacts/recommendation/week10_recommendation_examples.csv`
- `artifacts/recommendation/week10_data_alignment.csv`
- `artifacts/recommendation/week10_recommendation_summary.json`
- `artifacts/plots/week10/week10_metric_comparison.png`
- `artifacts/plots/week10/week10_median_rank.png`
- `artifacts/plots/week10/week10_user_level_distribution.png`
- `artifacts/plots/week10/week10_hit_at_10_by_user_level.png`
- `notebooks/08_week10_recommendation_ranking.ipynb`
