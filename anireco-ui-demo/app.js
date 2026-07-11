const DATA = window.ANIRECO_DATA || { catalog: [], product_rows: [], facets: {} };
const catalog = DATA.catalog || [];
const byId = new Map(catalog.map((item) => [Number(item.mal_id), item]));

const state = {
  source: "mal",
  profile: null,
  selectedTags: new Set(),
  selectedGenres: new Set(),
  coldGenres: new Set(),
  plan: new Map(),
  offsets: {},
};

const rowLabels = {
  general_recommendations: "For you right now",
  because_you_liked: "Because you liked...",
  continue_your_journey: "Continue your journey",
  people_you_like: "More from people and studios you like",
  give_it_a_try: "Give it a try",
};

const rowDescriptions = {
  general_recommendations: "Current/recent shows, popular high-score anchors, and the selected main ranker.",
  because_you_liked: "One recommendation per liked anchor first, then repeats only when it runs out.",
  continue_your_journey: "Sequels, side stories, parent stories, and franchise paths after prerequisite checks.",
  people_you_like: "Voice actors, directors, original creators, original story credits, and studios.",
  give_it_a_try: "Good titles outside the dominant taste sphere, filtered against tiny shorts and recap noise.",
};

const STANDALONE_FRANCHISE_ENTRY_IDS = new Set([
  80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 1215, 1917, 2273, 2581, 3927, 6288, 6336, 10808,
  10937, 19319, 23259, 24625, 31251, 31973, 33051, 35224, 37245, 37247, 37764, 37765, 40942, 44050, 49073, 49827,
  49828, 52168, 53199, 54902, 60449,
]);

const STANDALONE_FRANCHISE_PATTERNS = [
  /gundam\s*(00|seed|wing|age|x|unicorn|narrative|thunderbolt|build|witch|iron-blooded|ibo|reconguista|turn a|f91|victory)/i,
  /yu[\s-]*gi[\s-]*oh/i,
  /ookami to koushinryou.*merchant meets/i,
];

const GIVE_IT_A_TRY_CLASSIC_IDS = new Set([210, 1293, 435, 339, 467, 5081, 9253, 164, 457, 245, 329]);

const els = {
  onboarding: document.querySelector("#onboarding"),
  profilePanel: document.querySelector("#profilePanel"),
  profileTitle: document.querySelector("#profileTitle"),
  profileSubtitle: document.querySelector("#profileSubtitle"),
  profileStats: document.querySelector("#profileStats"),
  filterPanel: document.querySelector("#filterPanel"),
  rows: document.querySelector("#rows"),
  username: document.querySelector("#username"),
  usernameLabel: document.querySelector("#usernameLabel"),
  xmlUpload: document.querySelector("#xmlUpload"),
  coldFields: document.querySelector("#coldFields"),
  coldGenreChips: document.querySelector("#coldGenreChips"),
  advancedTagChips: document.querySelector("#advancedTagChips"),
  demoProfileButtons: document.querySelector("#demoProfileButtons"),
  importStatus: document.querySelector("#importStatus"),
  planDrawer: document.querySelector("#planDrawer"),
  drawerShade: document.querySelector("#drawerShade"),
  planItems: document.querySelector("#planItems"),
  planCount: document.querySelector("#planCount"),
  planSearch: document.querySelector("#planSearch"),
  planFilterType: document.querySelector("#planFilterType"),
  episodeValue: document.querySelector("#episodeValue"),
  episodeNumber: document.querySelector("#filterEpisodesNumber"),
  genreChips: document.querySelector("#genreChips"),
  clearTagFilters: document.querySelector("#clearTagFilters"),
  tagSearch: document.querySelector("#tagSearch"),
  selectedFilterChips: document.querySelector("#selectedFilterChips"),
};

function splitWeighted(value) {
  if (!value) return [];
  return String(value)
    .split("|")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const idx = part.lastIndexOf(":");
      if (idx < 0) return null;
      const left = part.slice(0, idx);
      const right = Number(part.slice(idx + 1));
      return [Number(left), Number.isFinite(right) ? right : 1];
    })
    .filter((edge) => edge && Number.isFinite(edge[0]));
}

function splitRelation(value) {
  if (!value) return [];
  return String(value)
    .split("|")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const idx = part.lastIndexOf(":");
      if (idx < 0) return null;
      const target = Number(part.slice(0, idx));
      const rel = part.slice(idx + 1);
      return { rel, target };
    })
    .filter((edge) => edge && Number.isFinite(edge.target));
}

function normalizeText(value) {
  return String(value || "").toLowerCase();
}

function normalizePersonKey(value) {
  return normalizeText(value).replace(/[^a-z0-9]+/g, "");
}

function westernToJapaneseKey(value) {
  const parts = String(value || "")
    .replaceAll(",", " ")
    .split(/\s+/)
    .filter(Boolean);
  if (parts.length < 2) return normalizePersonKey(value);
  return normalizePersonKey(`${parts.slice(1).join(" ")} ${parts[0]}`);
}

function canonicalPeopleNames(names) {
  const lookups = [...(DATA.lookups?.voice_actors || []), ...(DATA.lookups?.staff || [])];
  const byKey = new Map();
  for (const row of lookups) {
    if (!row.name) continue;
    byKey.set(normalizePersonKey(row.name), row.name);
    byKey.set(westernToJapaneseKey(row.name), row.name);
  }
  const output = [];
  const seen = new Set();
  for (const name of names || []) {
    const canonical = byKey.get(normalizePersonKey(name)) || byKey.get(westernToJapaneseKey(name)) || name;
    const key = normalizePersonKey(canonical);
    if (key && !seen.has(key)) {
      seen.add(key);
      output.push(canonical);
    }
  }
  return output;
}

function formatNumber(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "0";
  return new Intl.NumberFormat("en").format(Math.round(n));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function scoreSignal(item) {
  const score = Number(item.score || 0);
  const members = Math.log1p(Number(item.members || 0)) / Math.log1p(2_500_000);
  return score / 10 + 0.35 * members;
}

function classifyProfile(profile) {
  const count = profile.completedCount || profile.scored.size || profile.known.size;
  if (count < 50) return "Beginner";
  if (count < 150) return "Casual";
  if (count < 500) return "Fan";
  return "Veteran";
}

function profileFromLocalDemo() {
  const rows = DATA.product_rows || [];
  const demoIds = rows.flatMap((row) => row.items || []).map((item) => Number(item.mal_id));
  const known = new Set();
  const scored = new Map();
  catalog
    .filter((item) => Number(item.score || 0) >= 8 && Number(item.members || 0) > 200_000)
    .slice(0, 160)
    .forEach((item, index) => {
      if (index % 5 !== 0) {
        known.add(Number(item.mal_id));
        scored.set(Number(item.mal_id), 8 + (index % 3));
      }
    });
  demoIds.slice(0, 4).forEach((id) => known.delete(id));
  return buildProfile({ label: "Local demo profile", known, scored, favorites: new Set(), coldGenres: new Set() });
}

function buildProfile({
  label,
  known,
  scored,
  favorites,
  coldGenres,
  favoritePeopleNames = [],
  favoriteCharacterNames = [],
  favoriteStudios = [],
  favoriteVoiceActorIds = [],
  completedCountOverride = null,
  coldStart = false,
  levelOverride = null,
}) {
  const positive = [...scored.entries()]
    .filter(([, score]) => Number(score) >= 7)
    .map(([id]) => id)
    .filter((id) => byId.has(id));
  const profileItems = positive.map((id) => byId.get(id)).filter(Boolean);
  const genreCounts = new Map();
  const tagCounts = new Map();
  const studioCounts = new Map();
  const vaCounts = new Map();
  const staffCounts = new Map();
  const characterCounts = new Map();
  for (const item of profileItems) {
    const userScore = Number(scored.get(Number(item.mal_id)) || 7);
    const weight = Math.max(userScore - 6, 1) + (favorites?.has(Number(item.mal_id)) ? 3 : 0);
    for (const genre of item.genres || []) genreCounts.set(genre, (genreCounts.get(genre) || 0) + weight);
    for (const tag of item.tags || []) tagCounts.set(tag, (tagCounts.get(tag) || 0) + weight);
    for (const studio of item.studios || []) studioCounts.set(studio, (studioCounts.get(studio) || 0) + weight);
    for (const actor of item.voice_actors || []) vaCounts.set(actor, (vaCounts.get(actor) || 0) + weight);
    for (const staff of item.staff || []) staffCounts.set(staff, (staffCounts.get(staff) || 0) + weight);
    for (const character of String(item.characters || "").split("|").filter(Boolean)) {
      characterCounts.set(character, (characterCounts.get(character) || 0) + weight);
    }
  }
  for (const genre of coldGenres || []) genreCounts.set(genre, (genreCounts.get(genre) || 0) + 5);
  for (const studio of favoriteStudios || []) studioCounts.set(studio, (studioCounts.get(studio) || 0) + 8);
  for (const person of canonicalPeopleNames(favoritePeopleNames || [])) {
    vaCounts.set(person, (vaCounts.get(person) || 0) + 8);
    staffCounts.set(person, (staffCounts.get(person) || 0) + 8);
  }
  for (const character of favoriteCharacterNames || []) characterCounts.set(character, (characterCounts.get(character) || 0) + 8);
  for (const person of DATA.lookups?.voice_actors || []) {
    if ((person.ids || []).some((id) => favoriteVoiceActorIds.includes(id))) {
      vaCounts.set(person.name, (vaCounts.get(person.name) || 0) + 10);
    }
  }
  const profile = {
    label,
    known,
    scored,
    favorites: favorites || new Set(),
    favoritePeopleNames,
    favoriteCharacterNames,
    favoriteStudios,
    favoriteVoiceActorIds,
    positive: new Set(positive),
    completedCount: completedCountOverride ?? scored.size,
    genreCounts,
    tagCounts,
    studioCounts,
    vaCounts,
    staffCounts,
    characterCounts,
    coldStart,
    level: "",
  };
  profile.level = levelOverride || classifyProfile(profile);
  return profile;
}

function affinity(item, profile) {
  let score = 0;
  for (const genre of item.genres || []) score += (profile.genreCounts.get(genre) || 0) * 0.35;
  for (const tag of item.tags || []) score += (profile.tagCounts.get(tag) || 0) * 0.08;
  for (const studio of item.studios || []) score += (profile.studioCounts.get(studio) || 0) * 0.22;
  for (const actor of item.voice_actors || []) score += (profile.vaCounts.get(actor) || 0) * 0.18;
  for (const staff of item.staff || []) score += (profile.staffCounts.get(staff) || 0) * 0.18;
  if (profile.favorites.has(Number(item.mal_id))) score += 4;
  return score;
}

function isExplicit(item) {
  const text = `${item.rating || ""}|${(item.genres || []).join("|")}|${(item.tags || []).join("|")}`;
  return /hentai|erotica|rx/i.test(text);
}

function isWeakShort(item) {
  const minutes = Number(item.total_watch_minutes || 0);
  const members = Number(item.members || 0);
  return minutes > 0 && minutes < 35 && members < 30000;
}

function looksLikeRecap(item) {
  const title = normalizeText(item.title);
  const type = normalizeText(item.type);
  const minutes = Number(item.total_watch_minutes || 0);
  if (/\b(recap|recaps|digest|manner movie|soushuuhen)\b/.test(title)) return true;
  if (/\b(summary|beginning)\b/.test(title) && type !== "movie") return true;
  if (/\bsummary\b/.test(title) && minutes > 0 && minutes < 75) return true;
  return false;
}

function isFranchiseMovieOrNumberedEntry(item) {
  const title = normalizeText(item.title);
  if (/\bmovie\s*\d+\b/.test(title)) return true;
  if (/\b(movie|gekijouban)\b/.test(title) && splitRelation(item.relations).length) return true;
  if (/\bova\b/.test(title) || /\bspecials?\b/.test(title)) return true;
  return false;
}

function explicitIntent(profile = state.profile) {
  const selected = [...state.selectedTags, ...state.selectedGenres, ...(profile?.coldGenres || [])].map(normalizeText).join("|");
  const genreIntent = /hentai|erotica|rx|sexual/.test(selected);
  const toggleIntent = document.querySelector("#filterExplicit")?.checked && /hentai|erotica|rx/.test(selected);
  const profileIntent = profile && [...(profile.genreCounts || new Map()).keys()].some((label) => /hentai|erotica/i.test(label));
  return Boolean(genreIntent || toggleIntent || profileIntent);
}

function isStandaloneFranchiseEntry(item) {
  const malId = Number(item.mal_id);
  const title = String(item.title || item.title_english || "");
  if (STANDALONE_FRANCHISE_ENTRY_IDS.has(malId)) return true;
  return STANDALONE_FRANCHISE_PATTERNS.some((pattern) => pattern.test(title));
}

function franchiseAffinityCount(profile, item) {
  const key = franchiseKey(item);
  let count = 0;
  for (const id of profile.positive || []) {
    const seed = byId.get(Number(id));
    if (seed && franchiseKey(seed) === key) count += 1;
  }
  return count;
}

function looksLikeLaterEntry(item) {
  const title = normalizeText(item.title);
  if (isStandaloneFranchiseEntry(item)) return false;
  if (/\b(2nd|3rd|4th|5th|6th|7th|8th|9th)\b/.test(title)) return true;
  if (/\bseason\s*[2-9]\b/.test(title)) return true;
  if (/\bpart\s*[2-9]\b/.test(title)) return true;
  if (isFranchiseMovieOrNumberedEntry(item)) return true;
  if (["Special", "OVA"].includes(item.type)) return true;
  return splitRelation(item.relations).some((edge) => ["Prequel", "Parent Story", "Full Story"].includes(edge.rel));
}

function isLowValueContinuation(item) {
  const type = String(item.type || "");
  const minutes = Number(item.total_watch_minutes || 0);
  const members = Number(item.members || 0);
  if (isFutureEntry(item) || looksLikeRecap(item) || isWeakShort(item)) return true;
  if (isStandaloneFranchiseEntry(item)) return false;
  if (type === "Special" && (minutes < 75 || members < 120_000)) return true;
  if (type === "OVA" && minutes < 90 && members < 80_000) return true;
  if (isFranchiseMovieOrNumberedEntry(item) && type !== "Movie") return true;
  return false;
}

function seasonIndex(item) {
  const seasons = ["Winter", "Spring", "Summer", "Fall"];
  const year = Number(item.aired_year || 0);
  const rawSeason = String(item.season || "").trim();
  const season = rawSeason ? rawSeason[0].toUpperCase() + rawSeason.slice(1).toLowerCase() : "";
  const idx = seasons.indexOf(season);
  if (!year || idx < 0) return null;
  return year * 4 + idx;
}

function currentSeasonIndex() {
  if (currentSeasonIndex.cached !== undefined) return currentSeasonIndex.cached;
  const airingIndexes = catalog
    .map((item) => (/airing/i.test(String(item.status || "")) ? seasonIndex(item) : null))
    .filter((idx) => idx !== null);
  if (airingIndexes.length) {
    currentSeasonIndex.cached = Math.max(...airingIndexes);
    return currentSeasonIndex.cached;
  }
  const month = new Date().getMonth() + 1;
  const seasonIdx = month <= 3 ? 0 : month <= 6 ? 1 : month <= 9 ? 2 : 3;
  currentSeasonIndex.cached = new Date().getFullYear() * 4 + seasonIdx;
  return currentSeasonIndex.cached;
}

function isFutureEntry(item) {
  if (/not yet aired/i.test(String(item.status || ""))) return true;
  const idx = seasonIndex(item);
  if (idx !== null && idx > currentSeasonIndex()) return true;
  const year = Number(item.aired_year || 0);
  return year > new Date().getFullYear();
}

function currentWindow(item) {
  const idx = seasonIndex(item);
  const current = currentSeasonIndex();
  if (idx !== null) return idx <= current && idx >= current - 2;
  return String(item.status || "").includes("Airing") && !isFutureEntry(item);
}

function franchiseKey(item) {
  let title = normalizeText(item.title_english || item.title);
  title = title
    .replace(/\([^)]*\)/g, " ")
    .replace(/\b(the\s+)?movie\s*\d*\b/g, " ")
    .replace(/\b(ova|specials?|recaps?|recap|gekijouban)\b/g, " ")
    .replace(/\b(2nd|3rd|4th|5th|6th|7th|8th|9th|second|third|fourth)\b/g, " ")
    .replace(/\bseason\s*\d+\b/g, " ")
    .replace(/\bpart\s*\d+\b/g, " ")
    .replace(/[:\-].*$/g, " ")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
  const tokens = title.split(/\s+/).filter((token) => token.length > 1);
  return tokens.slice(0, 3).join(" ") || String(item.mal_id);
}

function mergeUniqueEntries(groups, limit = 12, capFranchise = true) {
  const seenIds = new Set();
  const seenFranchises = new Set();
  const output = [];
  const overflow = [];
  for (const group of groups) {
    for (const entry of group) {
      const malId = Number(entry.item?.mal_id);
      if (!malId || seenIds.has(malId)) continue;
      const key = franchiseKey(entry.item);
      if (capFranchise && seenFranchises.has(key)) {
        overflow.push(entry);
        continue;
      }
      seenIds.add(malId);
      seenFranchises.add(key);
      output.push(entry);
      if (output.length >= limit) return output;
    }
  }
  for (const entry of overflow) {
    const malId = Number(entry.item?.mal_id);
    if (!malId || seenIds.has(malId)) continue;
    seenIds.add(malId);
    output.push(entry);
    if (output.length >= limit) break;
  }
  return output;
}

function takeSegment(entries, limit, selectedIds, selectedFranchises, capFranchise = true) {
  const picked = [];
  const overflow = [];
  for (const entry of entries) {
    const malId = Number(entry.item?.mal_id);
    if (!malId || selectedIds.has(malId)) continue;
    const key = franchiseKey(entry.item);
    if (capFranchise && selectedFranchises.has(key)) {
      overflow.push(entry);
      continue;
    }
    selectedIds.add(malId);
    selectedFranchises.add(key);
    picked.push(entry);
    if (picked.length >= limit) return picked;
  }
  for (const entry of overflow) {
    const malId = Number(entry.item?.mal_id);
    if (!malId || selectedIds.has(malId)) continue;
    selectedIds.add(malId);
    picked.push(entry);
    if (picked.length >= limit) break;
  }
  return picked;
}

function activeFilters() {
  const selectedGenres = new Set(state.selectedGenres);
  return {
    search: normalizeText(document.querySelector("#filterSearch").value),
    genres: selectedGenres,
    yearMin: Number(document.querySelector("#filterYearMin").value || 0),
    yearMax: Number(document.querySelector("#filterYearMax").value || 9999),
    season: document.querySelector("#filterSeason").value,
    type: document.querySelector("#filterType").value,
    status: document.querySelector("#filterStatus").value,
    rating: document.querySelector("#filterRating").value,
    maxEpisodes: Number(document.querySelector("#filterEpisodes").value),
    allowExplicit: document.querySelector("#filterExplicit").checked,
    tags: new Set(state.selectedTags),
  };
}

function filterAllowsExplicit(filters) {
  const selected = [...(filters.tags || []), ...(filters.genres || []), filters.rating].map(normalizeText).join("|");
  return Boolean(filters.allowExplicit || /rx|hentai|erotica|sexual/.test(selected));
}

function isAdultGenreLabel(label) {
  return /hentai|erotica|rx/i.test(String(label || ""));
}

function itemMatchesGenreBoundary(item, genre) {
  if (isAdultGenreLabel(genre)) return isExplicit(item);
  return (item.genres || []).includes(genre) || (item.tags || []).includes(genre);
}

function selectedTagIntersectionIsSparse(filters) {
  if (!filterAllowsExplicit(filters) || !filters.tags?.size) return false;
  const cacheKey = JSON.stringify({
    genres: [...filters.genres].sort(),
    tags: [...filters.tags].sort(),
    rating: filters.rating,
    type: filters.type,
    season: filters.season,
    status: filters.status,
    yearMin: filters.yearMin,
    yearMax: filters.yearMax,
  });
  selectedTagIntersectionIsSparse.cache ||= new Map();
  if (selectedTagIntersectionIsSparse.cache.has(cacheKey)) return selectedTagIntersectionIsSparse.cache.get(cacheKey);
  let count = 0;
  for (const item of catalog) {
    if (state.plan.has(Number(item.mal_id))) continue;
    if (!filterAllowsExplicit(filters) && isExplicit(item)) continue;
    if (filters.genres?.size && ![...filters.genres].every((genre) => itemMatchesGenreBoundary(item, genre))) continue;
    const year = Number(item.aired_year || 0);
    if (year && (year < filters.yearMin || year > filters.yearMax)) continue;
    if (filters.season && String(item.season || "") !== filters.season) continue;
    if (filters.type && String(item.type || "") !== filters.type) continue;
    if (filters.status && String(item.status || "") !== filters.status) continue;
    if (filters.rating && String(item.rating || "") !== filters.rating) continue;
    if (filters.maxEpisodes > 0 && Number(item.episodes || 0) > filters.maxEpisodes) continue;
    if ([...filters.tags].every((tag) => (item.tags || []).includes(tag) || (item.genres || []).includes(tag))) count += 1;
    if (count >= 12) break;
  }
  const sparse = count === 0;
  selectedTagIntersectionIsSparse.cache.set(cacheKey, sparse);
  return sparse;
}

function boundaryBoost(item, filters) {
  let boost = 0;
  for (const genre of filters.genres || []) {
    if (itemMatchesGenreBoundary(item, genre)) boost += isAdultGenreLabel(genre) ? 0.45 : 0.55;
  }
  for (const tag of filters.tags || []) {
    if ((item.tags || []).includes(tag) || (item.genres || []).includes(tag)) boost += 0.9;
  }
  if (exactSearchMatch(item, filters.search)) boost += 5;
  return boost;
}

function searchMatches(item, search) {
  if (!search) return true;
  const id = String(item.mal_id || "");
  const haystack = normalizeText(
    `${item.mal_id} ${item.anilist_id || ""} ${item.title} ${item.title_english || ""} ${(item.studios || []).join(" ")} ${(item.genres || []).join(" ")} ${(item.tags || []).join(" ")}`,
  );
  return id === search || haystack.includes(search);
}

function exactSearchMatch(item, search) {
  if (!search) return false;
  const normalizedTitle = normalizeText(item.title);
  const normalizedEnglish = normalizeText(item.title_english);
  return String(item.mal_id || "") === search || normalizedTitle === search || normalizedEnglish === search;
}

function matchesUserBoundary(item, filters) {
  if (state.plan.has(Number(item.mal_id))) return false;
  if (!searchMatches(item, filters.search)) return false;
  if (filters.genres?.size) {
    for (const genre of filters.genres) {
      if (!itemMatchesGenreBoundary(item, genre)) return false;
    }
  }
  const year = Number(item.aired_year || 0);
  if (year && (year < filters.yearMin || year > filters.yearMax)) return false;
  if (filters.season && String(item.season || "") !== filters.season) return false;
  if (filters.type && String(item.type || "") !== filters.type) return false;
  if (filters.status && String(item.status || "") !== filters.status) return false;
  if (filters.rating && String(item.rating || "") !== filters.rating) return false;
  if (!filterAllowsExplicit(filters) && isExplicit(item)) return false;
  if (filters.maxEpisodes > 0 && Number(item.episodes || 0) > filters.maxEpisodes) return false;
  const relaxTags = selectedTagIntersectionIsSparse(filters);
  if (!relaxTags) {
    for (const tag of filters.tags) {
      if (!(item.tags || []).includes(tag) && !(item.genres || []).includes(tag)) return false;
    }
  }
  return true;
}

function passesFilters(item, filters) {
  return matchesUserBoundary(item, filters);
}

function candidateBase(profile, filters = activeFilters()) {
  const adultIntent = explicitIntent(profile) || filterAllowsExplicit(filters);
  return catalog.filter((item) => {
    const exactSearch = exactSearchMatch(item, filters.search);
    const adultFormat = adultIntent && isExplicit(item);
    if (profile.known.has(Number(item.mal_id))) return false;
    if (!matchesUserBoundary(item, filters)) return false;
    if (!exactSearch && looksLikeRecap(item)) return false;
    if (isFutureEntry(item)) return false;
    if (!exactSearch && !adultFormat && isWeakShort(item)) return false;
    if (!exactSearch && !adultFormat && looksLikeLaterEntry(item)) return false;
    return true;
  });
}

function profileSeedItems(profile, limit = 80) {
  const favoriteIds = [...profile.favorites].filter((id) => profile.positive.has(Number(id)));
  const scoredIds = [...profile.positive].sort(
    (a, b) => Number(profile.scored.get(b) || 0) - Number(profile.scored.get(a) || 0),
  );
  const ordered = [...favoriteIds, ...scoredIds];
  const seenIds = new Set();
  const seenFranchises = new Set();
  const firstPass = [];
  const overflow = [];
  for (const id of ordered) {
    const item = byId.get(Number(id));
    if (!item || seenIds.has(Number(id))) continue;
    seenIds.add(Number(id));
    const key = franchiseKey(item);
    if (seenFranchises.has(key)) overflow.push(item);
    else {
      seenFranchises.add(key);
      firstPass.push(item);
    }
  }
  return [...firstPass, ...overflow].slice(0, limit);
}

function recommenderCandidates(profile, filters = activeFilters(), limit = 80) {
  const weighted = new Map();
  for (const seed of profileSeedItems(profile, 100)) {
    const seedScore = Number(profile.scored.get(Number(seed.mal_id)) || 7);
    const seedBoost = profile.favorites.has(Number(seed.mal_id)) ? 1.3 : 1;
    for (const [target, weight] of splitWeighted(seed.recommendations)) {
      const item = byId.get(target);
      if (!item || profile.known.has(target)) continue;
      if (!matchesUserBoundary(item, filters)) continue;
      const adultFormat = (explicitIntent(profile) || filterAllowsExplicit(filters)) && isExplicit(item);
      if (isFutureEntry(item) || looksLikeRecap(item)) continue;
      if (!adultFormat && (looksLikeLaterEntry(item) || isWeakShort(item))) continue;
      const value =
        Number(weight) * seedBoost * (0.8 + Math.max(seedScore - 6, 1) * 0.12) + affinity(item, profile) + boundaryBoost(item, filters);
      const previous = weighted.get(target);
      if (!previous || value > previous.score) {
        weighted.set(target, {
          item,
          score: value,
          reason: `recommended from ${seed.title}`,
          anchor: seed.title,
        });
      }
    }
  }
  return [...weighted.values()].sort((a, b) => b.score - a.score).slice(0, limit);
}

function seasonalVarietyScore(item, profile) {
  const genreFit = (item.genres || []).reduce((total, genre) => total + (profile.genreCounts.get(genre) || 0), 0);
  const tagFit = (item.tags || []).reduce((total, tag) => total + (profile.tagCounts.get(tag) || 0), 0);
  const hash = Math.abs(Math.sin((Number(item.mal_id) || 1) * 12.9898 + String(profile.label || "").length * 78.233));
  return 0.22 * Math.log1p(genreFit) + 0.05 * Math.log1p(tagFit) + 0.18 * hash;
}

function generalRow(profile, filters = activeFilters()) {
  const base = candidateBase(profile, filters);
  const exact = filters.search
    ? base
        .filter((item) => exactSearchMatch(item, filters.search))
        .map((item) => ({ item, score: 999 + scoreSignal(item), reason: "direct search match", anchor: "search" }))
    : [];
  const current = base
    .filter((item) => currentWindow(item))
    .map((item) => ({
      item,
      score: 1.25 * scoreSignal(item) + seasonalVarietyScore(item, profile) + 0.18 * affinity(item, profile) + boundaryBoost(item, filters) + 0.35,
      reason: "current or recent season with quality and profile fit",
      anchor: "current season window",
    }))
    .sort((a, b) => b.score - a.score);
  const popular = base
    .filter((item) => !currentWindow(item))
    .filter((item) => Number(item.members || 0) >= 80_000 || Number(item.score || 0) >= 8)
    .map((item) => ({
      item,
      score: 1.65 * scoreSignal(item) + 0.04 * affinity(item, profile) + boundaryBoost(item, filters),
      reason: "popular high-score anchor",
      anchor: "quality/popularity prior",
    }))
    .sort((a, b) => b.score - a.score);
  const reco = recommenderCandidates(profile, filters, 240).map((entry) => ({
    ...entry,
    score: entry.score + scoreSignal(entry.item) + boundaryBoost(entry.item, filters),
    reason: `profile recommender: ${entry.reason}`,
  }));
  const fallback = base
    .map((item) => ({
      item,
      score: scoreSignal(item) + 0.08 * affinity(item, profile) + boundaryBoost(item, filters),
      reason: "content fallback: profile fit and catalog quality",
      anchor: "profile fallback",
    }))
    .sort((a, b) => b.score - a.score);
  const selectedIds = new Set();
  const selectedFranchises = new Set();
  const selected = [
    ...takeSegment(exact, 12, selectedIds, selectedFranchises, false),
    ...takeSegment(current.slice(0, 32), 3, selectedIds, selectedFranchises),
    ...takeSegment(popular.slice(0, 80), 3, selectedIds, selectedFranchises),
    ...takeSegment([...reco, ...fallback], 6, selectedIds, selectedFranchises),
  ];
  if (selected.length < 12) {
    selected.push(...takeSegment([...current, ...popular, ...reco, ...fallback], 12 - selected.length, selectedIds, selectedFranchises));
  }
  const extras = takeSegment([...exact, ...current, ...popular, ...reco, ...fallback], 180, selectedIds, selectedFranchises);
  return [...selected, ...extras];
}

function becauseYouLikedRow(profile, filters = activeFilters()) {
  const seeds = profileSeedItems(profile, 80);
  const chosen = new Set();
  const firstPass = [];
  const overflow = [];
  for (const seed of seeds) {
    const candidates = splitWeighted(seed.recommendations)
      .filter(([target]) => byId.has(target) && !profile.known.has(target))
      .map(([target, weight]) => {
        const item = byId.get(target);
        if (!matchesUserBoundary(item, filters)) return null;
        const adultFormat = (explicitIntent(profile) || filterAllowsExplicit(filters)) && isExplicit(item);
        if (isFutureEntry(item) || looksLikeRecap(item)) return null;
        if (!adultFormat && (looksLikeLaterEntry(item) || isWeakShort(item))) return null;
        return {
          item,
          score:
            Number(weight) * (profile.favorites.has(Number(seed.mal_id)) ? 1.25 : 1) * (1 + 0.04 * affinity(item, profile)) +
            boundaryBoost(item, filters),
          reason: `similar to ${seed.title}`,
          anchor: seed.title,
        };
      })
      .filter(Boolean)
      .sort((a, b) => b.score - a.score);
    const best = candidates.find((entry) => !chosen.has(Number(entry.item.mal_id)));
    if (best) {
      firstPass.push(best);
      chosen.add(Number(best.item.mal_id));
    }
    for (const entry of candidates.slice(1)) {
      if (!chosen.has(Number(entry.item.mal_id))) overflow.push(entry);
    }
  }
  return mergeUniqueEntries([firstPass, overflow.sort((a, b) => b.score - a.score)], 120);
}

function continueRow(profile, filters = activeFilters()) {
  const relationBoost = new Map([
    ["Sequel", 8],
    ["Prequel", 4],
    ["Parent Story", 4],
    ["Full Story", 4],
    ["Side Story", 2.3],
    ["Spin-Off", 1.8],
    ["Alternative Version", 2],
    ["Alternative Setting", 0.8],
    ["Other", 0.25],
  ]);
  const rows = [];
  for (const id of profile.positive) {
    const seed = byId.get(id);
    if (!seed) continue;
    for (const edge of splitRelation(seed.relations)) {
      const item = byId.get(edge.target);
      if (!item || profile.known.has(edge.target)) continue;
      if (!matchesUserBoundary(item, filters)) continue;
      if (!relationBoost.has(edge.rel)) continue;
      if (isLowValueContinuation(item)) continue;
      const franchiseCount = franchiseAffinityCount(profile, item);
      const standaloneBoost = isStandaloneFranchiseEntry(item) || franchiseCount >= 2 ? 1.2 : 0;
      rows.push({
        item,
        score: relationBoost.get(edge.rel) + standaloneBoost + scoreSignal(item) + boundaryBoost(item, filters),
        reason: `${edge.rel} connected to ${seed.title}`,
        anchor: seed.title,
      });
    }
  }
  const known = profile.known;
  for (const item of catalog) {
    const malId = Number(item.mal_id);
    if (known.has(malId) || isLowValueContinuation(item)) continue;
    if (!matchesUserBoundary(item, filters)) continue;
    for (const edge of splitRelation(item.relations)) {
      if (!known.has(edge.target)) continue;
      let boost = 0;
      if (["Prequel", "Parent Story", "Full Story"].includes(edge.rel)) boost = 7.5;
      else if (["Side Story", "Spin-Off", "Alternative Version"].includes(edge.rel)) boost = 2;
      else if (["Alternative Setting", "Other"].includes(edge.rel)) {
        const franchiseCount = franchiseAffinityCount(profile, item);
        boost = isStandaloneFranchiseEntry(item) || franchiseCount >= 2 ? 1.25 : 0.35;
      }
      if (!boost) continue;
      const anchorItem = byId.get(edge.target);
      rows.push({
        item,
        score: boost + scoreSignal(item) + boundaryBoost(item, filters),
        reason: `${edge.rel} connected to ${anchorItem?.title || "a known title"}`,
        anchor: anchorItem?.title || "known title",
      });
    }
  }
  const bestById = new Map();
  for (const row of rows) {
    const malId = Number(row.item.mal_id);
    if (!bestById.has(malId) || row.score > bestById.get(malId).score) bestById.set(malId, row);
  }
  return mergeUniqueEntries([[...bestById.values()].sort((a, b) => b.score - a.score)], 120);
}

function peopleRow(profile, filters = activeFilters()) {
  const studioKeys = [...profile.studioCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 18);
  const actorKeys = [...profile.vaCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 18);
  const staffKeys = [...profile.staffCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 18);
  const coldPeopleMode = profile.coldStart || (!studioKeys.length && !actorKeys.length && !staffKeys.length);
  const entries = candidateBase(profile, filters)
    .map((item) => {
      let score = coldPeopleMode ? scoreSignal(item) : 0;
      let anchor = "";
      for (const [studio, count] of studioKeys) {
        if ((item.studios || []).includes(studio)) {
          score += count * 0.8;
          anchor ||= studio;
        }
      }
      for (const [actor, count] of actorKeys) {
        if ((item.voice_actors || []).includes(actor)) {
          score += count * 0.55;
          anchor ||= actor;
        }
      }
      for (const [staff, count] of staffKeys) {
        if ((item.staff || []).includes(staff)) {
          score += count * 0.55;
          anchor ||= staff;
        }
      }
      return {
        item,
        score: score + scoreSignal(item) + boundaryBoost(item, filters),
        reason: anchor ? `shared people/studio signal: ${anchor}` : "popular studio and people signal",
        anchor,
      };
    })
    .filter((entry) => (coldPeopleMode ? Number(entry.item.members || 0) > 35_000 || Number(entry.item.score || 0) >= 7.3 : entry.score > 1.15))
    .sort((a, b) => b.score - a.score);
  return mergeUniqueEntries([entries], 120);
}

function giveItTryRow(profile, filters = activeFilters()) {
  const dominantGenres = new Set(
    [...profile.genreCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([genre]) => genre),
  );
  const adultIntent = explicitIntent(profile) || filterAllowsExplicit(filters);
  const entries = candidateBase(profile, filters)
    .filter((item) => adultIntent || !isExplicit(item))
    .filter((item) =>
      adultIntent
        ? Number(item.members || 0) > 800 || Number(item.score || 0) >= 5.8
        : Number(item.members || 0) > 12000 && Number(item.score || 0) >= 7.2,
    )
    .map((item) => {
      const overlap = (item.genres || []).filter((genre) => dominantGenres.has(genre)).length;
      const oldBonus = Number(item.aired_year || 3000) < 2012 ? 0.45 : 0;
      const genreShift = overlap === 0 ? 0.8 : overlap === 1 ? 0.35 : -0.4;
      return {
        item,
        score: scoreSignal(item) + genreShift + oldBonus - 0.025 * affinity(item, profile) + boundaryBoost(item, filters),
        reason: oldBonus ? "older acclaimed title outside the usual bubble" : "quality pick outside the dominant taste bubble",
        anchor: "controlled exploration",
      };
    })
    .sort((a, b) => b.score - a.score);
  const classics = [...GIVE_IT_A_TRY_CLASSIC_IDS]
    .map((id) => byId.get(id))
    .filter((item) => item && !profile.known.has(Number(item.mal_id)) && matchesUserBoundary(item, filters) && !looksLikeRecap(item))
    .map((item) => ({
      item,
      score: 3 + scoreSignal(item) + boundaryBoost(item, filters),
      reason: "classic/high-recognition entry point outside the usual bubble",
      anchor: "classic discovery",
    }));
  return mergeUniqueEntries([classics, entries], 120);
}

function localProductRows() {
  return (DATA.product_rows || []).map((row) => ({
    key: row.row,
    title: rowLabels[row.row] || row.row,
    description: rowDescriptions[row.row] || row.description || "",
    entries: (row.items || [])
      .map((entry) => {
        const item = byId.get(Number(entry.mal_id));
        return item
          ? {
              item,
              score: Number(entry.score || 0),
              reason: entry.reason || "",
              anchor: entry.anchor || "",
            }
          : null;
      })
      .filter(Boolean),
  }));
}

function titleForRow(key, profile) {
  if (key === "people_you_like" && profile?.coldStart) return "Popular people and Studios";
  return rowLabels[key] || key;
}

function descriptionForRow(key, profile) {
  if (key === "people_you_like" && profile?.coldStart) {
    return "Popular studios, voice actors, and staff signals from the catalog.";
  }
  return rowDescriptions[key] || "";
}

function generatedRows(profile, filters = activeFilters()) {
  return [
    {
      key: "general_recommendations",
      title: titleForRow("general_recommendations", profile),
      description: descriptionForRow("general_recommendations", profile),
      entries: generalRow(profile, filters),
    },
    {
      key: "because_you_liked",
      title: titleForRow("because_you_liked", profile),
      description: descriptionForRow("because_you_liked", profile),
      entries: becauseYouLikedRow(profile, filters),
    },
    {
      key: "continue_your_journey",
      title: titleForRow("continue_your_journey", profile),
      description: descriptionForRow("continue_your_journey", profile),
      entries: continueRow(profile, filters),
    },
    {
      key: "people_you_like",
      title: titleForRow("people_you_like", profile),
      description: descriptionForRow("people_you_like", profile),
      entries: peopleRow(profile, filters),
    },
    {
      key: "give_it_a_try",
      title: titleForRow("give_it_a_try", profile),
      description: descriptionForRow("give_it_a_try", profile),
      entries: giveItTryRow(profile, filters),
    },
  ];
}

function filteredEntries(row, filters) {
  const offset = state.offsets[row.key] || 0;
  const eligible = row.entries.filter(({ item }) => passesFilters(item, filters));
  const safeOffset = offset >= eligible.length ? 0 : offset;
  if (safeOffset !== offset) state.offsets[row.key] = 0;
  return eligible.slice(safeOffset, safeOffset + 12);
}

function renderRows() {
  if (!state.profile) return;
  const filters = activeFilters();
  const rows = state.profile.isLocalDemo ? localProductRows() : generatedRows(state.profile, filters);
  els.rows.innerHTML = "";
  for (const row of rows) {
    const eligibleTotal = row.entries.filter(({ item }) => passesFilters(item, filters)).length;
    const entries = filteredEntries(row, filters);
    if (!entries.length && row.key !== "continue_your_journey") continue;
    const section = document.createElement("section");
    section.className = "rec-row";
    const canRefresh = eligibleTotal > 12;
    section.innerHTML = `
      <div class="row-head">
        <div>
          <h2>${row.title}</h2>
          <p>${row.description}</p>
        </div>
        <button class="row-action" type="button" data-reroll="${row.key}" ${canRefresh ? "" : "disabled"}>${
          canRefresh ? "Refresh" : "No more"
        }</button>
      </div>
      <div class="carousel-shell">
        <button class="shelf-arrow shelf-arrow-left" type="button" data-scroll-row="${row.key}" data-direction="-1" aria-label="Scroll ${row.title} left">‹</button>
        <div class="carousel"></div>
        <button class="shelf-arrow shelf-arrow-right" type="button" data-scroll-row="${row.key}" data-direction="1" aria-label="Scroll ${row.title} right">›</button>
      </div>
    `;
    const carousel = section.querySelector(".carousel");
    if (entries.length) {
      for (const entry of entries) carousel.appendChild(card(entry));
    } else {
      const empty = document.createElement("p");
      empty.className = "empty-row row-empty-note";
      empty.textContent =
        "No continuation entries match right now. This can happen when sequels are already known, prerequisites are missing, or filters are too strict.";
      carousel.appendChild(empty);
    }
    enableCarouselDrag(carousel);
    els.rows.appendChild(section);
  }
  if (!els.rows.children.length) {
    els.rows.innerHTML = `<p class="empty-row">No rows match the current filters. Loosen the filters or refresh the profile.</p>`;
  }
}

function enableCarouselDrag(carousel) {
  let isDown = false;
  let startX = 0;
  let scrollLeft = 0;
  carousel.addEventListener("pointerdown", (event) => {
    isDown = true;
    carousel.classList.add("dragging");
    startX = event.clientX;
    scrollLeft = carousel.scrollLeft;
    carousel.setPointerCapture(event.pointerId);
  });
  carousel.addEventListener("pointermove", (event) => {
    if (!isDown) return;
    event.preventDefault();
    carousel.scrollLeft = scrollLeft - (event.clientX - startX);
  });
  for (const eventName of ["pointerup", "pointercancel", "pointerleave"]) {
    carousel.addEventListener(eventName, () => {
      isDown = false;
      carousel.classList.remove("dragging");
    });
  }
}

function card(entry) {
  const item = entry.item;
  const node = document.createElement("article");
  node.className = "anime-card";
  const anilist = item.anilist_id ? `https://anilist.co/anime/${item.anilist_id}` : "";
  node.innerHTML = `
    <img class="poster" src="${item.image_url || ""}" alt="" loading="lazy" />
    <div class="card-body">
      <div class="card-title">${item.title}</div>
      <div class="meta">
        <span class="pill">${item.type || "Anime"}</span>
        <span class="pill">${item.aired_year || "?"}</span>
        <span class="pill">Score ${item.score || "?"}</span>
      </div>
      <p class="reason">${entry.reason || (entry.anchor ? `anchored on ${entry.anchor}` : "profile match")}</p>
      <div class="card-actions">
        <button class="card-action" data-add="${item.mal_id}" type="button">Add</button>
        <a class="card-action link" href="${item.mal_url || anilist}" target="_blank" rel="noreferrer">Open</a>
      </div>
    </div>
  `;
  return node;
}

function renderProfile() {
  const p = state.profile;
  els.profilePanel.classList.remove("hidden");
  els.filterPanel.classList.remove("hidden");
  els.profileTitle.innerHTML = `${escapeHtml(p.label)} <span class="level-badge">${escapeHtml(p.level)}</span>`;
  els.profileSubtitle.textContent =
    p.level === "Beginner"
      ? "Safe recommendations lean on popular, high-score entry points."
      : p.level === "Veteran"
        ? "Veteran mode leans harder on discovery, people/studios, and graph paths."
        : "Recommendations combine similar anime, graph relations, and quality signals.";

  const topGenres = topKeys(p.genreCounts, 8);
  const topPeople = topKeys(mergeCounters(p.vaCounts, p.staffCounts), 24);
  const topCharacters = topKeys(p.characterCounts, 24);
  const topStudios = topKeys(p.studioCounts, 24);
  const favoriteAnime = favoriteAnimeNames(p);

  els.profileStats.innerHTML = `
    <div class="metric-strip">
      ${metricCard("Known", formatNumber(p.known.size))}
      ${metricCard("Scored", formatNumber(p.scored.size))}
      ${metricCard("Positive", formatNumber(p.positive.size))}
      ${metricCard("Mode", p.level)}
    </div>
    <div class="profile-summary-grid">
      ${summaryCard("Top genres", topGenres)}
      ${summaryCard("Favorite anime", favoriteAnime)}
      ${summaryCard("People / VA", topPeople)}
      ${summaryCard("Characters", topCharacters)}
      ${summaryCard("Studios", topStudios)}
    </div>
  `;
}

function topKeys(counter, limit = 5) {
  return [...counter.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([key]) => key)
    .filter(Boolean);
}

function mergeCounters(...counters) {
  const merged = new Map();
  for (const counter of counters) {
    for (const [key, value] of counter.entries()) merged.set(key, (merged.get(key) || 0) + value);
  }
  return merged;
}

function metricCard(label, value) {
  return `<div class="stat metric-card"><b>${escapeHtml(value)}</b><span>${escapeHtml(label)}</span></div>`;
}

function summaryCard(label, values) {
  const content = values.length
    ? values.map((value) => `<span class="profile-pill">${escapeHtml(value)}</span>`).join("")
    : `<span class="profile-empty">none yet</span>`;
  return `<div class="profile-summary-card"><span>${escapeHtml(label)}</span><div class="profile-pills">${content}</div></div>`;
}

function favoriteAnimeNames(profile) {
  return [...profile.favorites]
    .map((id) => byId.get(Number(id))?.title)
    .filter(Boolean);
}

function updatePlan() {
  els.planCount.textContent = String(state.plan.size);
  const allItems = [...state.plan.values()];
  const planTypes = [...new Set(allItems.map((item) => item.type).filter(Boolean))].sort();
  const previousType = els.planFilterType.value;
  els.planFilterType.innerHTML = `<option value="">All formats</option>${planTypes
    .map((type) => `<option>${type}</option>`)
    .join("")}`;
  if (planTypes.includes(previousType)) els.planFilterType.value = previousType;
  const search = normalizeText(els.planSearch.value);
  const type = els.planFilterType.value;
  const visible = allItems.filter((item) => {
    if (type && item.type !== type) return false;
    if (search && !normalizeText(`${item.title} ${item.title_english || ""}`).includes(search)) return false;
    return true;
  });
  if (!state.plan.size) {
    els.planItems.innerHTML = `<p class="empty-row">No anime selected yet.</p>`;
    return;
  }
  if (!visible.length) {
    els.planItems.innerHTML = `<p class="empty-row">No selected anime match the cart filters.</p>`;
    return;
  }
  els.planItems.innerHTML = "";
  for (const item of visible) {
    const node = document.createElement("div");
    node.className = "plan-item";
    node.innerHTML = `
      <img src="${item.image_url || ""}" alt="" />
      <div><b>${item.title}</b><p>${item.type || ""} - ${item.aired_year || ""}</p></div>
      <button class="secondary" data-remove="${item.mal_id}" type="button">Remove</button>
    `;
    els.planItems.appendChild(node);
  }
}

function fillSelect(id, values, label = "Any") {
  const select = document.querySelector(id);
  select.innerHTML = `<option value="">${label}</option>` + values.map((value) => `<option>${value}</option>`).join("");
}

function initFilters() {
  fillSelect("#filterSeason", DATA.facets.seasons || []);
  fillSelect("#filterType", DATA.facets.types || []);
  fillSelect("#filterRating", DATA.facets.ratings || []);
  fillSelect("#filterStatus", [...new Set(catalog.map((item) => item.status).filter(Boolean))].sort());
  const years = [...new Set(catalog.map((item) => item.aired_year).filter(Boolean))]
    .map(Number)
    .filter((year) => year > 1900)
    .sort((a, b) => b - a);
  document.querySelector("#filterYearMin").placeholder = String(Math.min(...years));
  document.querySelector("#filterYearMax").placeholder = String(Math.max(...years));

  renderGenreChips();
  renderTagCategories();
  renderSelectedFilters();
  els.coldGenreChips.innerHTML = (DATA.facets.genres || [])
    .map((tag) => `<button class="chip" data-cold="${tag}" type="button">${tag}</button>`)
    .join("");
  els.demoProfileButtons.innerHTML = (DATA.demo_profiles || [])
    .map((profile) => `<button class="chip" data-demo="${profile.id}" type="button">${profile.label}</button>`)
    .join("");
}

function renderGenreChips() {
  const query = normalizeText(els.tagSearch?.value || "");
  els.genreChips.innerHTML = (DATA.facets.genres || [])
    .filter((genre) => !query || normalizeText(genre).includes(query))
    .map((genre) => {
      const active = state.selectedGenres.has(genre) ? " active" : "";
      return `<button class="chip${active}" data-genre="${escapeHtml(genre)}" type="button">${escapeHtml(genre)}</button>`;
    })
    .join("");
}

function renderSelectedFilters() {
  const chips = [
    ...[...state.selectedGenres].map((label) => ({ label, kind: "genre" })),
    ...[...state.selectedTags].map((label) => ({ label, kind: "tag" })),
  ];
  els.selectedFilterChips.innerHTML = chips
    .map(
      (chip) =>
        `<button class="chip active" data-remove-filter="${escapeHtml(chip.kind)}:${escapeHtml(chip.label)}" type="button">${escapeHtml(
          chip.label,
        )} ×</button>`,
    )
    .join("");
}

function renderTagCategories() {
  const allowExplicit = document.querySelector("#filterExplicit")?.checked;
  const query = normalizeText(els.tagSearch?.value || "");
  const categories = DATA.facets.tag_categories || {};
  if (!allowExplicit && categories["Sexual Content"]) {
    const sexualTags = new Set(categories["Sexual Content"].map((tag) => tag.name));
    for (const tag of sexualTags) state.selectedTags.delete(tag);
  }
  els.advancedTagChips.innerHTML = Object.entries(categories)
    .filter(([category]) => allowExplicit || category !== "Sexual Content")
    .map(([category, tags]) => {
      const chips = tags
        .filter((tag) => !query || normalizeText(`${tag.name} ${tag.description || ""}`).includes(query))
        .map((tag) => {
          const active = state.selectedTags.has(tag.name) ? " active" : "";
          return `<button class="chip${active}" data-tag="${escapeHtml(tag.name)}" title="${escapeHtml(
            tag.description || "",
          )}" type="button">${escapeHtml(tag.name)}</button>`;
        })
        .join("");
      return chips ? `<div class="tag-category"><h3>${escapeHtml(category)}</h3><div class="chip-grid">${chips}</div></div>` : "";
    })
    .join("");
  renderSelectedFilters();
}

async function parseXmlFile(file) {
  if (!file) return { known: new Set(), scored: new Map(), favorites: new Set() };
  const text = await file.text();
  const doc = new DOMParser().parseFromString(text, "text/xml");
  const known = new Set();
  const scored = new Map();
  const favorites = new Set();
  doc.querySelectorAll("anime").forEach((node) => {
    const id = Number(node.querySelector("series_animedb_id")?.textContent || 0);
    const score = Number(node.querySelector("my_score")?.textContent || 0);
    const status = normalizeText(node.querySelector("my_status")?.textContent || "");
    if (!id || !byId.has(id)) return;
    if (status.includes("plan to watch") || status.includes("plantowatch") || status.includes("plan_to_watch")) return;
    known.add(id);
    if (score > 0) scored.set(id, score);
    if (score >= 10) favorites.add(id);
    if (status.includes("dropped") && score < 7) known.add(id);
  });
  return { known, scored, favorites };
}

async function fetchJikanFavorites(username) {
  const empty = { animeIds: new Set(), peopleNames: [], characterNames: [], studioNames: [] };
  if (!username) return empty;
  const response = await fetch(`https://api.jikan.moe/v4/users/${encodeURIComponent(username)}/favorites`);
  if (!response.ok) throw new Error(`Jikan favorites failed: ${response.status}`);
  const payload = await response.json();
  const ids = new Set();
  for (const item of payload?.data?.anime || []) {
    if (byId.has(Number(item.mal_id))) ids.add(Number(item.mal_id));
  }
  const peopleNames = (payload?.data?.people || []).map((item) => item.name).filter(Boolean);
  const characterNames = (payload?.data?.characters || []).map((item) => item.name).filter(Boolean);
  return { animeIds: ids, peopleNames: [...new Set(peopleNames)], characterNames: [...new Set(characterNames)], studioNames: [] };
}

async function fetchMalProfileFavorites(username) {
  const response = await fetch(`https://myanimelist.net/profile/${encodeURIComponent(username)}`);
  if (!response.ok) throw new Error(`MAL profile fallback failed: ${response.status}`);
  const html = await response.text();
  const doc = new DOMParser().parseFromString(html, "text/html");
  const favorites = new Set();
  const peopleNames = new Set();
  const characterNames = new Set();
  const studioNames = new Set();
  doc.querySelectorAll("#anime_favorites a[href*='/anime/']").forEach((link) => {
    const id = Number(link.href.match(/\/anime\/(\d+)/)?.[1] || 0);
    if (byId.has(id)) favorites.add(id);
  });
  doc.querySelectorAll("#people_favorites a[href*='/people/']").forEach((link) => {
    peopleNames.add(link.getAttribute("title") || decodeProfileSlug(link.href.match(/\/people\/\d+\/([^/?#]+)/)?.[1] || ""));
  });
  doc.querySelectorAll("#character_favorites a[href*='/character/']").forEach((link) => {
    characterNames.add(link.getAttribute("title") || decodeProfileSlug(link.href.match(/\/character\/\d+\/([^/?#]+)/)?.[1] || ""));
  });
  doc.querySelectorAll("#company_favorites a[href*='/anime/producer/'], #company_favorites a[href*='/company/']").forEach((link) => {
    studioNames.add(link.getAttribute("title") || decodeProfileSlug(link.href.match(/\/(?:producer|company)\/\d+\/([^/?#]+)/)?.[1] || ""));
  });
  return { animeIds: favorites, peopleNames: [...peopleNames], characterNames: [...characterNames], studioNames: [...studioNames] };
}

function decodeProfileSlug(value) {
  try {
    return decodeURIComponent(String(value || "").replaceAll("_", " "));
  } catch {
    return String(value || "").replaceAll("_", " ");
  }
}

function termsFromInput(id) {
  return document
    .querySelector(id)
    .value.split(",")
    .map((part) => part.trim())
    .filter(Boolean);
}

function manualAnimeFavorites() {
  const ids = new Set();
  for (const term of termsFromInput("#manualAnime")) {
    const numeric = Number(term);
    if (Number.isFinite(numeric) && byId.has(numeric)) {
      ids.add(numeric);
      continue;
    }
    const query = normalizeText(term);
    const found = catalog.find(
      (item) => normalizeText(item.title) === query || normalizeText(item.title_english) === query,
    );
    if (found) ids.add(Number(found.mal_id));
  }
  return ids;
}

function lookupFavoriteNames(inputId, lookupKey) {
  const names = [];
  const rows = DATA.lookups?.[lookupKey] || [];
  for (const term of termsFromInput(inputId)) {
    const query = normalizeText(term);
    const found = rows.find((row) => normalizeText(row.name).includes(query) || query.includes(normalizeText(row.name)));
    names.push(found?.name || term);
  }
  return [...new Set(names)];
}

function manualStudios() {
  const studioSet = new Set(DATA.facets?.studios || []);
  return termsFromInput("#manualStudios").map((term) => {
    const query = normalizeText(term);
    return [...studioSet].find((studio) => normalizeText(studio).includes(query)) || term;
  });
}

async function fetchAniListProfile(username) {
  const query = `
    query($name:String){
      MediaListCollection(userName:$name,type:ANIME){
        lists{entries{score status media{idMal}}}
      }
      User(name:$name){
        favourites{
          anime{nodes{idMal}}
          characters{nodes{name{full}}}
          staff{nodes{name{full}}}
          studios{nodes{name}}
        }
      }
    }`;
  const response = await fetch("https://graphql.anilist.co", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ query, variables: { name: username } }),
  });
  if (!response.ok) throw new Error(`AniList import failed: ${response.status}`);
  const payload = await response.json();
  const known = new Set();
  const scored = new Map();
  const favorites = new Set();
  for (const list of payload?.data?.MediaListCollection?.lists || []) {
    for (const entry of list.entries || []) {
      const id = Number(entry?.media?.idMal || 0);
      if (!id || !byId.has(id)) continue;
      if (String(entry.status || "").toUpperCase() === "PLANNING") continue;
      known.add(id);
      if (Number(entry.score || 0) > 0) scored.set(id, Number(entry.score));
    }
  }
  for (const node of payload?.data?.User?.favourites?.anime?.nodes || []) {
    if (byId.has(Number(node.idMal))) favorites.add(Number(node.idMal));
  }
  const favoritePeopleNames = (payload?.data?.User?.favourites?.staff?.nodes || []).map((node) => node?.name?.full).filter(Boolean);
  const favoriteCharacterNames = (payload?.data?.User?.favourites?.characters?.nodes || []).map((node) => node?.name?.full).filter(Boolean);
  const favoriteStudios = (payload?.data?.User?.favourites?.studios?.nodes || []).map((node) => node?.name).filter(Boolean);
  return { known, scored, favorites, favoritePeopleNames, favoriteCharacterNames, favoriteStudios };
}

async function buildFromInputs() {
  els.importStatus.textContent = "Building profile...";
  let known = new Set();
  let scored = new Map();
  let favorites = new Set();
  try {
    const manualAnime = manualAnimeFavorites();
    const manualPeople = lookupFavoriteNames("#manualPeople", "voice_actors").concat(
      lookupFavoriteNames("#manualPeople", "staff"),
    );
    const manualCharacters = lookupFavoriteNames("#manualCharacters", "characters");
    const manualStudioNames = manualStudios();
    for (const id of manualAnime) {
      known.add(id);
      scored.set(id, 10);
    }
    if (state.source === "cold") {
      state.profile = buildProfile({
        label: "Cold-start profile",
        known,
        scored,
        favorites: manualAnime,
        coldGenres: state.coldGenres,
        favoritePeopleNames: manualPeople,
        favoriteCharacterNames: manualCharacters,
        favoriteStudios: manualStudioNames,
        coldStart: true,
      });
    } else if (state.source === "anilist") {
      const username = els.username.value.trim();
      if (!username) throw new Error("AniList username is required.");
      const imported = await fetchAniListProfile(username);
      imported.favorites = new Set([...imported.favorites, ...manualAnime]);
      for (const id of manualAnime) {
        imported.known.add(id);
        imported.scored.set(id, 10);
      }
      state.profile = buildProfile({
        label: `AniList: ${username}`,
        ...imported,
        coldGenres: new Set(),
        favoritePeopleNames: [...manualPeople, ...(imported.favoritePeopleNames || [])],
        favoriteCharacterNames: [...manualCharacters, ...(imported.favoriteCharacterNames || [])],
        favoriteStudios: [...manualStudioNames, ...(imported.favoriteStudios || [])],
      });
    } else {
      const xml = await parseXmlFile(els.xmlUpload.files[0]);
      known = xml.known;
      scored = xml.scored;
      favorites = new Set(xml.favorites || []);
      for (const id of manualAnime) {
        known.add(id);
        scored.set(id, 10);
      }
      const username = els.username.value.trim();
      let importedFavoritePeople = [];
      let importedFavoriteCharacters = [];
      let importedFavoriteStudios = [];
      if (username) {
        try {
          const importedFavorites = await fetchJikanFavorites(username);
          for (const id of importedFavorites.animeIds || []) favorites.add(id);
          importedFavoritePeople = importedFavorites.peopleNames || [];
          importedFavoriteCharacters = importedFavorites.characterNames || [];
          importedFavoriteStudios = importedFavorites.studioNames || [];
        } catch (error) {
          try {
            const importedFavorites = await fetchMalProfileFavorites(username);
            for (const id of importedFavorites.animeIds || []) favorites.add(id);
            importedFavoritePeople = importedFavorites.peopleNames || [];
            importedFavoriteCharacters = importedFavorites.characterNames || [];
            importedFavoriteStudios = importedFavorites.studioNames || [];
          } catch {
            els.importStatus.textContent = `${error.message}. Browser fallback may be blocked; continuing with XML/manual favorites.`;
          }
        }
      }
      for (const id of manualAnime) favorites.add(id);
      for (const id of favorites) {
        if (byId.has(id)) {
          known.add(id);
          if (!scored.has(id)) scored.set(id, 10);
        }
      }
      state.profile = buildProfile({
        label: username ? `MAL: ${username}` : "MAL XML import",
        known,
        scored,
        favorites,
        coldGenres: new Set(),
        favoritePeopleNames: [...manualPeople, ...importedFavoritePeople],
        favoriteCharacterNames: [...manualCharacters, ...importedFavoriteCharacters],
        favoriteStudios: [...manualStudioNames, ...importedFavoriteStudios],
      });
    }
    state.profile.isLocalDemo = false;
    if (!state.profile.known.size && !state.profile.scored.size && state.source !== "cold") {
      throw new Error("No catalog-matched list entries found. Upload XML or try the local demo profile.");
    }
    els.importStatus.textContent = "Profile ready.";
    renderProfile();
    renderRows();
  } catch (error) {
    els.importStatus.textContent = error.message || "Could not build profile.";
  }
}

function useLocalDemo() {
  const veteranDemo = (DATA.demo_profiles || []).find((profile) => profile.level === "Veteran") || (DATA.demo_profiles || [])[0];
  if (veteranDemo) {
    useDemoProfile(veteranDemo.id);
    els.importStatus.textContent = `Loaded ${veteranDemo.label} with the live row generator.`;
    return;
  }
  state.profile = profileFromLocalDemo();
  state.profile.isLocalDemo = false;
  els.importStatus.textContent = "Using a synthetic local profile with the live row generator.";
  renderProfile();
  renderRows();
}

function useDemoProfile(profileId) {
  const demo = (DATA.demo_profiles || []).find((profile) => profile.id === profileId);
  if (!demo) return;
  const known = new Set((demo.known_ids || []).map(Number).filter((id) => byId.has(id)));
  const scored = new Map();
  for (const row of demo.scored || []) {
    const id = Number(row.mal_id);
    if (byId.has(id)) scored.set(id, Number(row.score || 10));
  }
  for (const id of demo.favorite_anime_ids || []) {
    if (byId.has(Number(id))) {
      known.add(Number(id));
      if (!scored.has(Number(id))) scored.set(Number(id), 10);
    }
  }
  state.profile = buildProfile({
    label: demo.label,
    known,
    scored,
    favorites: new Set((demo.favorite_anime_ids || []).map(Number).filter((id) => byId.has(id))),
    coldGenres: new Set(),
    favoriteVoiceActorIds: (demo.favorite_voice_actor_ids || []).map(Number),
    completedCountOverride: Number(demo.completed_count || scored.size),
    levelOverride: demo.level || null,
  });
  state.profile.isLocalDemo = false;
  els.importStatus.textContent = `Loaded ${demo.label}.`;
  renderProfile();
  renderRows();
}

function setSource(source) {
  state.source = source;
  document.querySelectorAll(".source-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.source === source);
  });
  els.usernameLabel.textContent = source === "anilist" ? "AniList username" : "MAL username";
  els.username.placeholder = source === "anilist" ? "AniList username" : "MAL username";
  els.coldFields.classList.toggle("hidden", source !== "cold");
  els.username.closest(".field").classList.toggle("hidden", source === "cold");
  document.querySelector(".xml-field").classList.toggle("hidden", source !== "mal");
}

document.addEventListener("click", (event) => {
  const sourceButton = event.target.closest("[data-source]");
  if (sourceButton) setSource(sourceButton.dataset.source);

  const tagButton = event.target.closest("[data-tag]");
  if (tagButton) {
    const tag = tagButton.dataset.tag;
    if (state.selectedTags.has(tag)) state.selectedTags.delete(tag);
    else state.selectedTags.add(tag);
    tagButton.classList.toggle("active");
    renderSelectedFilters();
    renderRows();
  }

  const genreButton = event.target.closest("[data-genre]");
  if (genreButton) {
    const genre = genreButton.dataset.genre;
    if (state.selectedGenres.has(genre)) state.selectedGenres.delete(genre);
    else state.selectedGenres.add(genre);
    genreButton.classList.toggle("active");
    renderSelectedFilters();
    renderRows();
  }

  const removeFilterButton = event.target.closest("[data-remove-filter]");
  if (removeFilterButton) {
    const [kind, ...rest] = removeFilterButton.dataset.removeFilter.split(":");
    const label = rest.join(":");
    if (kind === "genre") state.selectedGenres.delete(label);
    if (kind === "tag") state.selectedTags.delete(label);
    renderGenreChips();
    renderTagCategories();
    renderRows();
  }

  const coldButton = event.target.closest("[data-cold]");
  if (coldButton) {
    const tag = coldButton.dataset.cold;
    if (state.coldGenres.has(tag)) state.coldGenres.delete(tag);
    else state.coldGenres.add(tag);
    coldButton.classList.toggle("active");
  }

  const demoButton = event.target.closest("[data-demo]");
  if (demoButton) {
    useDemoProfile(demoButton.dataset.demo);
  }

  const addButton = event.target.closest("[data-add]");
  if (addButton) {
    const item = byId.get(Number(addButton.dataset.add));
    if (item) state.plan.set(Number(item.mal_id), item);
    updatePlan();
    renderRows();
  }

  const removeButton = event.target.closest("[data-remove]");
  if (removeButton) {
    state.plan.delete(Number(removeButton.dataset.remove));
    updatePlan();
    renderRows();
  }

  const reroll = event.target.closest("[data-reroll]");
  if (reroll && state.profile && !reroll.disabled) {
    const key = reroll.dataset.reroll;
    const row = (state.profile.isLocalDemo ? localProductRows() : generatedRows(state.profile, activeFilters())).find((item) => item.key === key);
    const total = row ? row.entries.filter(({ item }) => passesFilters(item, activeFilters())).length : 0;
    if (total <= 12) return;
    const next = (state.offsets[key] || 0) + 12;
    state.offsets[key] = next >= total ? 0 : next;
    renderRows();
  }
  const shelfArrow = event.target.closest("[data-scroll-row]");
  if (shelfArrow) {
    const shell = shelfArrow.closest(".carousel-shell");
    const carousel = shell?.querySelector(".carousel");
    if (!carousel) return;
    const direction = Number(shelfArrow.dataset.direction || 1);
    const cardWidth = carousel.querySelector(".anime-card")?.getBoundingClientRect().width || 214;
    carousel.scrollBy({ left: direction * cardWidth * 3, behavior: "smooth" });
  }
});

document.querySelector("#buildProfile").addEventListener("click", buildFromInputs);
document.querySelector("#useLocalDemo").addEventListener("click", useLocalDemo);
els.clearTagFilters.addEventListener("click", () => {
  state.selectedTags.clear();
  state.selectedGenres.clear();
  document.querySelectorAll("[data-tag].active, [data-genre].active").forEach((button) => button.classList.remove("active"));
  renderGenreChips();
  renderTagCategories();
  renderRows();
});
document.querySelector("#openPlan").addEventListener("click", () => {
  els.planDrawer.classList.add("open");
  els.drawerShade.classList.add("open");
});
document.querySelector("#closePlan").addEventListener("click", () => {
  els.planDrawer.classList.remove("open");
  els.drawerShade.classList.remove("open");
});
document.querySelector("#drawerShade").addEventListener("click", () => {
  els.planDrawer.classList.remove("open");
  els.drawerShade.classList.remove("open");
});
document.querySelector("#clearPlan").addEventListener("click", () => {
  state.plan.clear();
  updatePlan();
  renderRows();
});
els.planSearch.addEventListener("input", updatePlan);
els.planFilterType.addEventListener("change", updatePlan);

document.querySelectorAll(".filters input, .filters select").forEach((input) => {
  input.addEventListener("input", () => {
    if (input.id === "filterEpisodes") els.episodeNumber.value = input.value;
    if (input.id === "filterEpisodesNumber") document.querySelector("#filterEpisodes").value = input.value || 0;
    const episodeLimit = Number(document.querySelector("#filterEpisodes").value || 0);
    els.episodeValue.textContent = episodeLimit > 0 ? String(episodeLimit) : "No limit";
    if (input.id === "filterExplicit" || input.id === "tagSearch") {
      renderGenreChips();
      renderTagCategories();
    }
    renderRows();
  });
  input.addEventListener("change", () => {
    if (input.id === "filterExplicit" || input.id === "tagSearch") {
      renderGenreChips();
      renderTagCategories();
    }
    renderRows();
  });
});

initFilters();
updatePlan();
setSource("mal");
useLocalDemo();
