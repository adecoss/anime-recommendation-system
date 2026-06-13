# Week 10 Recommendation, Ranking, and Evaluation Report

## Task Framing

The Week 10 task is personalized anime recommendation as a ranking problem. The unit being ranked is one anime catalog entry from `anime_dataset.csv`. Given a user profile made from scored liked anime, the system ranks candidate anime that are plausible next discoveries.

This is not plain score prediction. The product question is: **which unseen anime should be shown near the top of a user's recommendation list?** A correct offline recommendation means that a held-out anime the user liked is ranked highly among sampled alternatives.

The final product is level-aware. The profile bands use the requested known-entry thresholds: `0/20/50/100/250/500/750`. In the stable benchmark, profile depth is approximated by known liked ratings because the 2020 interaction file has no watch-status or timestamp fields. In the new current-rating collector, profile depth is being improved to use scored catalog-matched list rows with status kept as `completed`, `watching`, `on_hold`, `dropped`, or `plan_to_watch`.

| Profile band | Min known | Max known | Primary signal | Main risk |
| --- | --- | --- | --- | --- |
| Newcomer | 0 | 19.0 | cold-start onboarding, genre filters, popularity, score, short entry points | no reliable personal vector; collaborative filtering needs a popularity/content fallback |
| Casual Viewer | 20 | 49.0 | genre onboarding, recognizable popular titles, short lists, early content similarity | some watched anime exist, but the profile is still too small for a stable latent taste vector |
| Explorer | 50 | 99.0 | similar anime, relation navigation, popularity prior, high-rated anchors | taste is still forming; the model can overfit to a few early favorites |
| Regular Fan | 100 | 249.0 | collaborative filtering, relation continuation, content/tag similarity, current discovery | model should balance familiar taste with controlled exploration |
| Dedicated Fan | 250 | 499.0 | collaborative ranking, niche clusters, relation navigation, controlled novelty | the user knows many common recommendations, so repetition becomes more visible |
| Veteran Fan | 500 | 749.0 | collaborative ranking, niche clusters, graph-aware expansion, seasonal/current discovery | many obvious items are already known; recommendations need novelty and coverage |
| Completionist | 750 | + | long-tail discovery, novelty, obscure catalog coverage, graph-aware exploration | hardest group: exact recovery is difficult and obvious catalog items are saturated |

## Data Alignment

The Week 10 model connects the catalog and interaction layers required by the assignment.

| Layer | Rows | Definition |
| --- | --- | --- |
| catalog | 14965 | anime_dataset rows |
| positive_interactions | 41928689 | ratings >= 7 aligned to catalog |
| candidate_pool | 9197 | catalog anime with >= 20 positive interactions |
| model_users | 109857 | eligible users used for matrix factorization training |
| evaluation_users | 15000 | users with one positive item held out |

The interaction layer is currently `ratings_processed.csv` from the stable 2020 MAL Kaggle ratings dataset. That source is useful because it is large and reproducible, but it is not current. A new public-list collection pipeline is being built separately in `00_collect_current_user_ratings.ipynb`; it now keeps every scored catalog-matched anime with its list status, so a future version can avoid recommending anime a user dropped, is watching, or has on hold. That updated interaction layer is planned for a later deliverable and is not used in this Week 10 benchmark yet.

Positive feedback is defined as rating `>= 7`, which matches MAL's visible label of `Good`. The candidate pool is not the entire catalog: it is the set of catalog anime with at least 20 positive ratings after alignment. This avoids evaluating matrix factorization on titles with almost no collaborative evidence.

## User Controls and Cold Start Policy

The recommender should not behave like one undifferentiated score. Before ranking, the interface can apply product controls: genre include/exclude filters, demographic and content-rating filters, explicit-content toggles, score floors, episode count, total-watchtime limits, release year, and seasonal recency.

This follows the Week 8 and Week 9 recommendation logic. Content and metadata filters are most useful when interaction evidence is sparse. Collaborative filtering becomes useful once enough behavior exists. A hybrid ranker blends both intuitions by keeping a popularity prior while using personalized latent evidence.

For Newcomers, the system generates a separate entry-point pool rather than pretending that matrix factorization can personalize with no history. The pool filters out explicit entries and obvious continuation entries, then favors high-score, high-engagement, shorter or medium-length anime.

| Title | Type | Score | Members | Episodes | Genres | Entry score |
| --- | --- | --- | --- | --- | --- | --- |
| Sousou no Frieren | TV | 9.260 | 1456504 | 28.0 | Adventure / Award Winning / Drama / Fantasy / Action / Comedy / Romance | 0.780 |
| Steins;Gate | TV | 9.070 | 2822509 | 24.0 | Drama / Sci-Fi / Suspense / Comedy / Romance | 0.764 |
| Koe no Katachi | Movie | 8.930 | 2627641 | 1.0 | Award Winning / Drama / Romance | 0.749 |
| Kimi no Na wa. | Movie | 8.820 | 3037822 | 1.0 | Award Winning / Drama / Sci-Fi / Comedy / Romance / Slice of Life | 0.741 |
| Fullmetal Alchemist: Brotherhood | TV | 9.110 | 3696136 | 64.0 | Action / Adventure / Drama / Fantasy / Sci-Fi / Comedy | 0.736 |
| Shingeki no Kyojin | TV | 8.570 | 4377647 | 25.0 | Action / Award Winning / Drama / Suspense / Sci-Fi / Fantasy / Adventure / Horror | 0.715 |
| Vinland Saga | TV | 8.780 | 1833638 | 24.0 | Action / Adventure / Drama | 0.700 |
| Violet Evergarden | TV | 8.690 | 2005781 | 13.0 | Drama / Action / Sci-Fi / Romance / Slice of Life | 0.688 |
| Death Note | TV | 8.620 | 4314806 | 37.0 | Supernatural / Suspense / Fantasy / Romance / Mystery | 0.685 |
| One Punch Man | TV | 8.470 | 3530194 | 12.0 | Action / Comedy / Gourmet / Sci-Fi / Sports | 0.685 |

## Systems Compared

Four ranking systems are compared on the same candidate sets.

| System | Role | Why it matters |
| --- | --- | --- |
| `random_uniform` | chance floor | Shows what happens when rank order contains no useful signal. |
| `popularity_baseline` | fair simple baseline | Anime discovery is strongly popularity-driven, so this is the baseline the stronger model must beat. |
| `latent_svd` | stronger collaborative model | Learns user and anime factors from the sparse positive user-anime matrix. |
| `hybrid_svd_popularity` | product-safe blend | Combines normalized SVD score with popularity to reduce risk for sparse profiles. |

The hybrid score is a convex blend after normalizing each score vector over the candidate set. This matters because raw SVD scores and popularity counts are not on the same scale.

## Evaluation Protocol

For each eligible user, one liked anime is deterministically held out. The remaining liked anime form the user's profile and training evidence. The held-out pair is removed from the training matrix.

Each evaluation case ranks 1 held-out liked anime against 100 sampled catalog negatives. The metrics are Hit@10, NDCG@10, MRR, median rank, and mean rank. This protocol tests whether the system can recover a known liked item from plausible alternatives. Because the 2020 ratings file has no timestamps, this is not claimed as chronological next-watch prediction.

## Overall Results

| Method | Hit@10 | NDCG@10 | MRR | Median rank | Mean rank |
| --- | --- | --- | --- | --- | --- |
| latent_svd | 0.910 | 0.716 | 0.658 | 1.0 | 5.693 |
| hybrid_svd_popularity | 0.910 | 0.716 | 0.658 | 1.0 | 4.891 |
| popularity_baseline | 0.770 | 0.501 | 0.430 | 4.0 | 7.839 |
| random_uniform | 0.100 | 0.045 | 0.051 | 51.0 | 50.936 |

The result has the expected ladder: random is near chance, popularity is a strong simple baseline, and the personalized SVD models beat popularity clearly. `latent_svd` is the narrow winner by Hit@10 and NDCG@10, while the hybrid keeps a slightly lower mean rank. For a product, I would still keep the hybrid as the default because it is safer for sparse profiles and aligns with the cold-start strategy.

The main plots are:

- `artifacts/plots/week10/week10_metric_comparison.png`
- `artifacts/plots/week10/week10_discovery_metrics.png`
- `artifacts/plots/week10/week10_median_rank.png`
- `artifacts/plots/week10/week10_rank_distribution_by_method.png`
- `artifacts/plots/week10/week10_method_hit_by_user_level.png`


## Discovery-Quality Metrics

Accuracy alone can reward a recommender that keeps showing the same famous anime. I added three compact product metrics from each method's top-10 lists:

- `Coverage@10`: share of the candidate pool that appears at least once in top-10 recommendations.
- `Novelty@10`: average inverse popularity, where higher means less popularity-heavy.
- `Genre diversity@10`: average genre spread inside top-10 lists.

| Method | Coverage@10 | Unique recommended | Novelty@10 | Unique genres@10 | Genre diversity@10 |
| --- | --- | --- | --- | --- | --- |
| latent_svd | 0.518 | 4762 | 0.275 | 14.648 | 0.316 |
| hybrid_svd_popularity | 0.431 | 3967 | 0.257 | 14.766 | 0.313 |
| popularity_baseline | 0.203 | 1868 | 0.202 | 15.107 | 0.307 |
| random_uniform | 1.000 | 9197 | 0.578 | 13.217 | 0.390 |

These metrics should not be optimized alone. `random_uniform` has perfect coverage and high novelty because it is uncontrolled, but its Hit@10 is near chance. The useful interpretation is the tradeoff: SVD greatly improves accuracy over popularity while also expanding coverage; the hybrid sacrifices some coverage and novelty for a safer popularity prior.

## Results by User Level

| Profile band | Users | Median profile | Hybrid Hit@10 | Hybrid NDCG@10 | Median rank |
| --- | --- | --- | --- | --- | --- |
| Newcomer | 1726 | 11.0 | 0.880 | 0.717 | 1.0 |
| Casual Viewer | 2773 | 34.0 | 0.933 | 0.761 | 1.0 |
| Explorer | 3323 | 72.0 | 0.920 | 0.739 | 1.0 |
| Regular Fan | 4757 | 153.0 | 0.913 | 0.702 | 2.0 |
| Dedicated Fan | 1852 | 326.0 | 0.894 | 0.665 | 2.0 |
| Veteran Fan | 393 | 580.0 | 0.863 | 0.653 | 2.0 |
| Completionist | 176 | 920.5 | 0.801 | 0.584 | 2.0 |

The profile-band analysis is useful because it shows where the product policy should change. Newcomer and Casual Viewer users still benefit from popularity and safe entry points. Regular and Dedicated fans have enough history for collaborative ranking. Veteran and Completionist users are harder: they have already consumed many obvious titles, so the system needs novelty, graph-aware navigation, and long-tail coverage.

## MyList Demonstration

I also added a local demonstration using `data/raw/MyList.xml`. The script treats scored titles rated at least 7 as seed evidence, excluding `Plan to Watch` entries from the seed set, and excludes every anime already present in the list from recommendation candidates. The remaining catalog candidates are ranked with the same hybrid SVD-popularity score.

The parsed XML contains 1,338 rows and 588 liked scored catalog seeds. Under the profile bands, this profile is classified as `Completionist`.

The public Jikan profile for `Champux` was fetched as a small derived summary for this demonstration. It reports 1,023 completed anime, 3 watching, 5 on hold, 0 dropped, 308 planned, a mean score of 7.44, and 244.7 days watched. This agrees with the local `MyList.xml` export closely enough to treat the profile as a dense `Completionist` case.

| Title | Type | MAL score | Episodes | Genres | MyList hybrid |
| --- | --- | --- | --- | --- | --- |
| Mahou Shoujo Madoka★Magica Movie 2: Eien no Monogatari | Movie | 8.380 | 1 | Drama / Suspense / Action / Sci-Fi / Fantasy / Adventure / Horror / Comedy / Mystery | 0.653 |
| Mahou Shoujo Madoka★Magica Movie 1: Hajimari no Monogatari | Movie | 8.220 | 1 | Drama / Suspense / Action / Sci-Fi / Fantasy / Adventure / Horror / Comedy / Mystery | 0.650 |
| Sword Art Online Movie: Ordinal Scale | Movie | 7.560 | 1 | Action / Adventure / Sci-Fi | 0.568 |
| Nisekoi: | TV | 7.340 | 12 | Comedy / Romance | 0.553 |
| Toaru Majutsu no Index II | TV | 7.510 | 24 | Action / Fantasy / Sci-Fi / Girls Love / Comedy / Romance | 0.543 |
| Natsume Yuujinchou San | TV | 8.560 | 13 | Slice of Life / Supernatural / Fantasy / Comedy | 0.537 |
| Hinamatsuri | TV | 8.110 | 12 | Comedy / Action / Slice of Life | 0.530 |
| Natsume Yuujinchou Shi | TV | 8.630 | 13 | Slice of Life / Supernatural / Action / Fantasy / Horror / Comedy | 0.525 |
| Mondaiji-tachi ga Isekai kara Kuru Sou desu yo? | TV | 7.400 | 10 | Action / Comedy / Fantasy | 0.519 |
| Kuroshitsuji | TV | 7.650 | 24 | Action / Mystery / Supernatural / Gourmet / Boys Love / Sci-Fi / Fantasy / Horror / Suspense / Comedy / R | 0.514 |
| Boku no Hero Academia 4th Season | TV | 7.860 | 25 | Action | 0.505 |
| Mimi wo Sumaseba | Movie | 8.230 | 1 | Drama / Romance / Fantasy / Comedy / Slice of Life / Sports | 0.498 |


The naive table is intentionally left in the report because it is useful failure analysis. It shows that a pure positive-seed ranker can surface candidates that are numerically close to the profile but operationally wrong: SAO after low SAO scores, `Nisekoi:` before `Nisekoi`, second seasons with missing prerequisites, and OVA/special side content.

For comparison, I added a guarded MyList output as a Week 12 preview. It keeps the hybrid score, but applies graph/status rules after ranking:

- block summaries or recaps of known titles
- block candidates with missing prerequisites
- block candidates whose prerequisite is only `Plan to Watch`
- block candidates whose prerequisite was rated below 7
- block side/special content unless the parent prerequisite is liked

| Naive rank | Blocked title | Type | Reason |
| --- | --- | --- | --- |
| 1 | Mahou Shoujo Madoka★Magica Movie 2: Eien no Monogatari | Movie | summary/recap of known title 9756  /  missing prerequisite 11977 |
| 2 | Mahou Shoujo Madoka★Magica Movie 1: Hajimari no Monogatari | Movie | summary/recap of known title 9756 |
| 3 | Sword Art Online Movie: Ordinal Scale | Movie | low-rated prerequisite 21881:4 |
| 4 | Nisekoi: | TV | prerequisite is only plan-to-watch 18897 |
| 8 | Natsume Yuujinchou Shi | TV | missing prerequisite 10379 |
| 15 | Nisekoi OVA | OVA | prerequisite is only plan-to-watch 18897  /  side/special content without liked parent |
| 18 | Nanatsu no Taizai: Imashime no Fukkatsu | TV | missing prerequisite 31722 |
| 20 | Sword Art Online: Alicization - War of Underworld | TV | prerequisite is only plan-to-watch 36474 |
| 22 | Bungou Stray Dogs 2nd Season | TV | missing prerequisite 31478 |
| 24 | Mahoutsukai no Yome: Hoshi Matsu Hito | OVA | prerequisite is only plan-to-watch 35062  /  side/special content without liked parent |


The guardrail block summary is:

| Guardrail category | Blocked count | Blocked pct of reviewed |
| --- | --- | --- |
| summary_or_recap | 5 | 0.055 |
| missing_prerequisite | 55 | 0.604 |
| plan_to_watch_prerequisite | 19 | 0.209 |
| low_rated_prerequisite | 8 | 0.088 |
| side_content_without_liked_parent | 24 | 0.264 |
| any_block | 91 | 0.455 |

After those guardrails, the corrected top recommendations are:

| Guarded rank | Title | Type | MAL score | Hybrid score |
| --- | --- | --- | --- | --- |
| 1 | Toaru Majutsu no Index II | TV | 7.510 | 0.543 |
| 2 | Natsume Yuujinchou San | TV | 8.560 | 0.537 |
| 3 | Hinamatsuri | TV | 8.110 | 0.530 |
| 4 | Mondaiji-tachi ga Isekai kara Kuru Sou desu yo? | TV | 7.400 | 0.519 |
| 5 | Kuroshitsuji | TV | 7.650 | 0.514 |
| 6 | Boku no Hero Academia 4th Season | TV | 7.860 | 0.505 |
| 7 | Mimi wo Sumaseba | Movie | 8.230 | 0.498 |
| 8 | Shoujo Shuumatsu Ryokou | TV | 8.240 | 0.496 |
| 9 | Asobi Asobase | TV | 8.190 | 0.494 |
| 10 | Tsumiki no Ie | Movie | 7.990 | 0.489 |
| 11 | Kurenai no Buta | Movie | 7.970 | 0.485 |
| 12 | Hellsing | TV | 7.500 | 0.483 |

The corrected list is not meant to replace the Week 10 benchmark. It demonstrates the next architectural step: use the graph layer and list-status evidence as guardrails around the ranking model.

This example is not part of the offline metric because it has no held-out label. It is a qualitative product test: does the system return plausible unseen titles after excluding watched, dropped, on-hold, watching, and planned entries?

## Error Analysis

Strong cases happen when the user's remaining liked anime form a coherent collaborative neighborhood: shared genres, franchises, demographic patterns, or audience taste communities. The model is especially strong when the held-out item belongs to a dense part of the interaction matrix.

Failure cases are expected when:

- the user profile is sparse or broad
- the held-out title represents a secondary interest rather than the user's dominant taste
- a sequel or side story should be surfaced by relation navigation instead of global discovery ranking
- a user has already consumed most obvious recommendations
- the sampled-negative setup is easier than full-catalog ranking
- the 2020 ratings distribution no longer matches current anime discourse
- the interaction layer does not yet know whether older ratings came from completed, dropped, watching, or on-hold entries

The most important limitation is not model complexity. It is evaluation validity: without timestamps, the benchmark cannot prove next-season or next-watch prediction. It proves only that the model can recover hidden liked items under a controlled ranking setup.

## Rubric Alignment

| Week 10 requirement | Evidence in this deliverable |
| --- | --- |
| Baseline system | `random_uniform` and `popularity_baseline`; popularity is the fair baseline for anime discovery. |
| Stronger system | `latent_svd` and `hybrid_svd_popularity` matrix-factorization ranking. |
| Offline evaluation report | Candidate pool, train/test holdout, metrics, and baseline comparison are explicit. |
| Error analysis | Strong cases, failure cases, timestamp limitation, sparse profile risk, dropped/watching/on-hold caveat, and sequel/navigation caveats are discussed. |
| Task framing | The system is recommendation/ranking, not score prediction. |
| Data alignment | Catalog, interaction layer, candidate pool, model users, and evaluation users are documented. |

## Reproducible Artifacts

Run:

```bash
python src/08_run_week10_recommendation.py
```

Main outputs:

- `artifacts/recommendation/week10_evaluation_metrics.csv`
- `artifacts/recommendation/week10_metrics_by_user_level.csv`
- `artifacts/recommendation/week10_beginner_entrypoint_candidates.csv`
- `artifacts/recommendation/week10_mylist_recommendation_example.csv`
- `artifacts/recommendation/week10_mylist_guardrail_comparison.csv`
- `artifacts/recommendation/week10_mylist_guardrail_block_summary.csv`
- `artifacts/recommendation/week10_mylist_guarded_recommendation_example.csv`
- `artifacts/recommendation/week10_champux_profile_summary.json`
- `artifacts/recommendation/week10_recommendation_examples.csv`
- `artifacts/recommendation/week10_data_alignment.csv`
- `artifacts/recommendation/week10_recommendation_summary.json`
- `artifacts/plots/week10/week10_metric_comparison.png`
- `artifacts/plots/week10/week10_discovery_metrics.png`
- `artifacts/plots/week10/week10_median_rank.png`
- `artifacts/plots/week10/week10_rank_distribution_by_method.png`
- `artifacts/plots/week10/week10_method_hit_by_user_level.png`
- `artifacts/plots/week10/week10_mylist_top_recommendations.png`
- `artifacts/plots/week10/week10_mylist_guarded_top_recommendations.png`
- `notebooks/08_week10_recommendation_ranking.ipynb`
