# AniReco UI Demo

Static local prototype for the anime recommendation product.

## Run Locally

The app is static. From the project root:

```bash
python anireco-ui-demo/tools/build_demo_data.py
python -m http.server 5174 -d anireco-ui-demo
```

Then open:

```text
http://localhost:5174
```

Opening `index.html` directly also works in many browsers because the data is
loaded from `assets/anireco-data.js`, not fetched as JSON.

## Current Behavior

- Starts with a local demo profile generated from the current artifacts.
- Lets the user choose MAL, AniList, or cold start.
- MAL mode accepts an optional XML export and optional username for public
  favorites. The UI warns that username-only MAL access may not return the full
  list.
- If the MAL favorites call fails, the UI attempts a profile-page fallback; if
  browser CORS blocks that too, users can enter favorite anime, people,
  characters, and studios manually.
- AniList mode tries public GraphQL list/favorites import by username.
- Cold start asks for genre preferences and avoids obvious sequels, later
  seasons, franchise movies, specials, and OVAs as starter picks.
- Includes four collected local demo profiles, one per user category, with
  favorites from the collected profile features. Personal XML imports stay
  manual and are not bundled as demo users.
- Shows a profile summary and category: Beginner, Casual, Fan, or Veteran.
- Shows favorite anime, people/VA, characters, and studios when known.
- Renders streaming-service-style rows:
  - For you right now
  - Because you liked...
  - Continue your journey
  - More from people and studios you like
  - Give it a try
- User controls define the recommendation candidate pool first, then each row
  ranks inside that pool. Searching an exact title or MAL id prioritizes that
  title when it is not already known.
- Year filtering is a min/max range.
- Episode filtering supports slider and typed input; `0` means no limit.
- Genres and tags are multi-selectable, with a clear-selected control. Tags
  are grouped by AniList tag category plus a small fallback category for
  catalog tags without a reference category.
- Row refresh changes the visible window for that row.
- Adding a show to Plan to Watch removes it from rows and stores it in the
  searchable/filterable drawer for the current browser session.

## Move to Separate Product Repo

When the product repo is created, this folder can become the repo root. Keep the
data builder as a local export step until there is a backend/API.
