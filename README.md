# Anime Discovery Recommender

Personal anime recommendation project built around a raw-source catalog, public user-rating interactions, graph navigation, and people/character metadata. The goal is not a single ranked list, but a streaming-service-style recommendation surface with multiple rows: general picks, similar titles, franchise continuation, voice actor/staff discoveries, and controlled exploration outside the user's comfort zone.

## Product Aim

The final recommender should support four profile bands:

- `Beginner` (`1-49` known anime): needs popular, short, high-confidence entry points.
- `Casual` (`50-149` known anime): has enough history for simple personalization; similar anime and direct relations become useful.
- `Fan` (`150-499` known anime): has a stronger taste profile; recommendations can mix known preferences with current shows and controlled genre expansion.
- `Veteran` (`500+` known anime): has seen many obvious titles; novelty, coverage, graph navigation, and people-based discovery matter more.

Cold-start users are handled separately: ask for preferred genres/tags, show recognizable popular titles, then build starter rows from selected interests.

## Repository Structure

```text
data/
  raw_sources/      MAL/Jikan, AniList, and AniDB source caches
  reference/        static reference metadata such as AniList tag descriptions
  processed/        final catalog, user ratings, cast/staff edge tables
  build/            checkpoints, retry registries, build summaries, audits

notebooks/
  00_collect_current_user_ratings.ipynb
  01_gather_raw_sources.ipynb
  02_build_anime_dataset.ipynb
  03_catalog_eda.ipynb
  04_improve_anime_dataset.ipynb
  05_project_foundation_and_schema.ipynb
  06_catalog_representation.ipynb
  07_catalog_segmentation.ipynb
  08_recommender_evaluation.ipynb
  09_graph_discovery.ipynb
  10_recommendation_product_rows.ipynb

src/
  00_collect_current_user_ratings.py
  01_gather_raw_sources.py
  02_build_anime_dataset.py
  03_run_catalog_eda.py
  04_improve_anime_dataset.py
  05_build_catalog_features.py
  06_analyze_catalog_representation.py
  07_segment_catalog.py
  08_evaluate_recommenders.py
  09_build_discovery_graph.py
  10_build_recommendation_product_rows.py
```

## Data Sources

- MyAnimeList/Jikan: catalog metadata, user-facing scores, popularity, relations, recommendations, characters, and staff/person identifiers.
- AniList: genre/tag system, tag weights, current airing status, compact media metadata, character/staff favorite counts, and extra recommendations.
- AniDB/Shoko: fallback metadata, episode/duration fixes, production-origin tags, explicit/loli fallback tags, and relation edge support.
- Public MAL lists: anonymized, scored interactions collected through a local encrypted username queue.

## Rebuild Pipeline

Run the core catalog pipeline in order:

```bash
python src/01_gather_raw_sources.py --jikan --anidb --refresh-shoko --retry-failed
python src/02_build_anime_dataset.py
python src/03_run_catalog_eda.py
python src/04_improve_anime_dataset.py
python src/03_run_catalog_eda.py
```

Build representation, segmentation, graph, and recommender artifacts:

```bash
python src/05_build_catalog_features.py
python src/06_analyze_catalog_representation.py
python src/07_segment_catalog.py
python src/09_build_discovery_graph.py
python src/08_evaluate_recommenders.py
python src/10_build_recommendation_product_rows.py
```

Collect current public user interactions separately:

```bash
python src/00_collect_current_user_ratings.py --discover
python src/00_collect_current_user_ratings.py --collect
```

The user collector keeps usernames encrypted only while they are pending retry. Successful users are written as anonymized ids and the plaintext username is not retained.

## Recommendation Surface

The final product layer generates row-based outputs with at most 12 titles per row:

- `general_recommendations`: profile-aware hybrid recommendations with popularity/score/current-season priors.
- `because_you_liked`: item-to-item recommendations anchored on highly rated or favorite titles.
- `continue_your_journey`: sequels, parent stories, side stories, specials, and franchise relations that pass sequencing guardrails.
- `people_you_like`: anime connected through favorite voice actors, directors, original creators, or original character designers.
- `give_it_a_try`: controlled exploration outside the user's dominant genres/tags.

Optional controls define the candidate search space before each row ranks titles: search text/id, year range, genres/tags, format, airing status, rating, hentai toggle, and episode count. Within those boundaries, each row uses its own recommender logic instead of merely hiding items after a fixed list is produced.

## Important Outputs

```text
data/processed/anime_dataset.csv
data/processed/current_user_ratings.csv
data/processed/current_user_profile_features.csv
data/processed/anime_voice_actor_edges.csv
data/processed/anime_staff_edges.csv
artifacts/recommendation/
artifacts/graph/
artifacts/plots/
```

Large raw caches and generated matrices are intentionally kept local or ignored where appropriate.
