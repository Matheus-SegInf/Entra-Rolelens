/**
 * Entra RoleLens — self-contained Azure App Service server.
 *
 * Serves the full RoleLens API (6 routes) from a LOCAL data file
 * (data/master.json) instead of Cloudflare D1 + KV. The keyword + BM25
 * search index is rebuilt in memory at startup, faithfully porting:
 *   - pipeline/push_to_cloudflare.py  (index build: keywords, tf, idf, doc_length)
 *   - worker/src/index.ts             (route logic + ranking)
 *
 * No cloud dependencies. Everything runs from the committed repo.
 */

'use strict';

const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 8080;

// ===========================================================================
// Constants (ported verbatim from worker/src/index.ts and push_to_cloudflare.py)
// ===========================================================================

// Query-time stop words (worker extractKeywords)
const STOP_WORDS = new Set([
  'a', 'an', 'the', 'to', 'for', 'of', 'in', 'and', 'or', 'with', 'how',
  'can', 'i', 'my', 'is', 'are', 'do', 'does', 'what', 'which', 'who',
]);

// Index-time stop words (push_to_cloudflare.py STOP_WORDS)
const PY_STOP_WORDS = new Set([
  'a', 'an', 'the', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'of',
  'or', 'and', 'but', 'not', 'with', 'by', 'from', 'that', 'this', 'all',
  'as', 'be', 'can', 'do', 'has', 'have', 'it', 'its', 'no', 'so', 'up',
  'was', 'will', 'if', 'how', 'when', 'where', 'which', 'who', 'you',
  'your', 'via', 'using', 'into', 'their', 'any', 'each',
]);

// Verbs that appear in almost every task — excluded from topic matching
const GENERIC_VERBS = new Set([
  'configure', 'manage', 'update', 'create', 'read', 'view', 'set', 'add',
  'remove', 'delete', 'enable', 'disable', 'get', 'list', 'show', 'use',
  'make', 'change', 'edit', 'modify', 'access', 'allow', 'block',
]);

// Clusters of related feature areas for affinity scoring
const RELATED_AREAS = {
  'Security - Authentication methods': [
    'Authentication', 'Temporary Access Pass',
    'Multi-factor authentication', 'Password Reset', 'Identity Protection',
  ],
  'Agent Identity': [
    'Enterprise applications', 'Application management',
  ],
  'Backup and Recovery': [
    'Directory', 'Identity Governance',
  ],
  'Tenant Governance': [
    'External collaboration', 'Cross-tenant access',
  ],
  'Privileged Identity Management': [
    'Roles and administrators', 'Identity Governance',
  ],
  'Conditional Access': [
    'Security - Authentication methods', 'Identity Protection', 'Named locations',
  ],
};

// Smoothed Okapi BM25 (k1=1.2, b=0.5)
const BM25_K1 = 1.2;
const BM25_B = 0.5;

const PERM_WRITE_VERBS = new Set([
  'create', 'delete', 'update', 'manage', 'assign', 'write', 'set',
  'add', 'remove', 'enable', 'disable', 'reset', 'configure', 'allProperties',
]);

// ===========================================================================
// Tokenizers
// ===========================================================================

// Query tokenizer — worker extractKeywords()
function extractKeywords(q) {
  return [
    ...new Set(
      String(q)
        .toLowerCase()
        .split(/[\s\p{P}]+/u)
        .map((w) => w.trim())
        .filter((w) => w.length >= 2 && !STOP_WORDS.has(w))
    ),
  ];
}

// Index tokenizer — push_to_cloudflare.py extract_keywords() (dedup)
function pyExtractKeywords(text) {
  const words = String(text).toLowerCase().match(/[a-z]{3,}/g) || [];
  return [...new Set(words.filter((w) => !PY_STOP_WORDS.has(w)))];
}

// Index tokenizer with repetition — push_to_cloudflare.py extract_keywords_with_repetition()
function pyExtractKeywordsRepeat(text) {
  const words = String(text).toLowerCase().match(/[a-z]{3,}/g) || [];
  return words.filter((w) => !PY_STOP_WORDS.has(w));
}

// ===========================================================================
// Pure scoring helpers (ported from worker)
// ===========================================================================

function privilegeFactor(permCount, isPrivileged) {
  let factor;
  if (permCount <= 20) factor = 1.4;
  else if (permCount <= 50) factor = 1.2;
  else if (permCount <= 100) factor = 1.0;
  else if (permCount <= 200) factor = 0.8;
  else factor = 0.6;
  if (isPrivileged) factor *= 0.85;
  return factor;
}

function affinityFactor(area, dominantArea) {
  if (area === dominantArea) return 2.0;
  const related = RELATED_AREAS[dominantArea];
  if (related) {
    return related.includes(area) ? 1.5 : 0.6;
  }
  return 0.5;
}

function splitCamelCase(s) {
  return String(s)
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .toLowerCase()
    .split(/[^a-z]+/)
    .filter(Boolean);
}

function rankRelevantPermissions(queryTokens, permissions) {
  if (!permissions || permissions.length === 0) return [];

  const scored = permissions.map((perm) => {
    const segments = perm.split('/').slice(1); // drop namespace prefix
    const permWords = segments.flatMap((seg) => splitCamelCase(seg));

    let score = 0;
    for (const token of queryTokens) {
      if (permWords.includes(token)) score += 10;
      else if (permWords.some((w) => w.startsWith(token))) score += 5;
      else if (permWords.some((w) => w.includes(token))) score += 2;
    }

    const lastWord = splitCamelCase(segments[segments.length - 1] || '').pop() || '';
    if (PERM_WRITE_VERBS.has(lastWord)) score *= 1.2;

    return { perm, score };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, 5).map((s) => s.perm);
}

function generateMatchReasoning(srcQuery, task, matchType) {
  const tokens = String(srcQuery)
    .toLowerCase()
    .split(/[\s\p{P}]+/u)
    .map((w) => w.trim())
    .filter((w) => w.length >= 2 && !STOP_WORDS.has(w));
  if (tokens.length === 0) return null;

  const taskWords = new Set(
    String(task).toLowerCase().split(/[\s\p{P}]+/u).map((w) => w.trim()).filter(Boolean)
  );
  const matched = tokens.filter((t) => taskWords.has(t));
  if (matched.length === 0) return null;

  return matchType === 'exact'
    ? `matched: ${matched[0]} (exact)${matched.length > 1 ? ', ' + matched.slice(1).join(', ') : ''}`
    : `matched: ${matched.join(', ')}`;
}

// ===========================================================================
// In-memory database + search index
// ===========================================================================

const DATA_PATH = path.join(__dirname, 'data', 'master.json');

const DB = {
  raw: null,
  roles: [],                 // raw role objects from master.json
  rolesById: new Map(),      // id -> role
  rolesByNameLower: new Map(), // lower(displayName) -> role
  tasks: [],                 // normalized task objects (with synthetic id)
  tasksById: new Map(),      // id -> task
  kwIndex: new Map(),        // keyword -> [{ taskId, weight, tf, idf }]
  corpus: { total_docs: 0, avg_doc_length: 6.0, idf: new Map() },
  status: {},
};

function loadData() {
  const raw = JSON.parse(fs.readFileSync(DATA_PATH, 'utf8'));
  DB.raw = raw;
  DB.roles = raw.roles || [];
  DB.tasks = [];
  DB.rolesById = new Map();
  DB.rolesByNameLower = new Map();
  DB.tasksById = new Map();
  DB.kwIndex = new Map();

  for (const r of DB.roles) {
    DB.rolesById.set(r.id, r);
    DB.rolesByNameLower.set(String(r.displayName).toLowerCase(), r);
  }

  // Normalize tasks + assign synthetic integer ids.
  // alt_roles in master.json are display NAMES — resolve to ids (mirrors
  // push_to_cloudflare.py resolve_guids) so enrichment matches worker output.
  const rawTasks = raw.tasks || [];
  rawTasks.forEach((t, i) => {
    const altIds = (t.alt_roles || [])
      .map((name) => {
        const role = DB.rolesByNameLower.get(String(name).toLowerCase());
        return role ? role.id : null;
      })
      .filter(Boolean);

    const task = {
      id: i,
      task_description: t.task || '',
      feature_area: t.feature_area || '',
      min_role_id: t.role_id || null,
      alt_role_ids: altIds,
      source_url: t.source_url || '',
      out_of_scope: t.out_of_scope != null ? t.out_of_scope : null,
      doc_length: 0,
    };
    DB.tasks.push(task);
    DB.tasksById.set(task.id, task);
  });

  buildSearchIndex();
  buildStatus();

  console.log(
    `Loaded ${DB.roles.length} roles, ${DB.tasks.length} tasks, ` +
    `${DB.corpus.idf.size} unique keywords (avg_doc_length=${DB.corpus.avg_doc_length.toFixed(2)})`
  );
}

// Build BM25 corpus stats + the keyword inverted index.
// Mirrors compute_bm25_stats() and push_task_search() from the pipeline.
function buildSearchIndex() {
  const tfPerTask = new Map();   // taskId -> Map(keyword -> tf)
  const docsContaining = new Map(); // keyword -> df
  let totalLen = 0;

  for (const task of DB.tasks) {
    const text = task.task_description + ' ' + task.feature_area;
    const tokens = pyExtractKeywordsRepeat(text);

    const tf = new Map();
    for (const tok of tokens) tf.set(tok, (tf.get(tok) || 0) + 1);
    tfPerTask.set(task.id, tf);
    task.doc_length = tokens.length;
    totalLen += tokens.length;

    for (const kw of tf.keys()) {
      docsContaining.set(kw, (docsContaining.get(kw) || 0) + 1);
    }
  }

  const N = DB.tasks.length;
  DB.corpus.total_docs = N;
  DB.corpus.avg_doc_length = N > 0 ? totalLen / N : 0;

  // IDF (smoothed Okapi): ln((N - df + 0.5) / (df + 0.5) + 1)
  DB.corpus.idf = new Map();
  for (const [kw, df] of docsContaining) {
    DB.corpus.idf.set(kw, Math.log((N - df + 0.5) / (df + 0.5) + 1));
  }

  // task_search rows: description keywords weight 1.0, feature-area keywords 0.5
  const addRow = (kw, taskId, weight) => {
    const tf = tfPerTask.get(taskId).get(kw) || 1.0;
    const idf = DB.corpus.idf.get(kw) || 0.0;
    if (!DB.kwIndex.has(kw)) DB.kwIndex.set(kw, []);
    DB.kwIndex.get(kw).push({ taskId, weight, tf, idf });
  };

  for (const task of DB.tasks) {
    for (const kw of pyExtractKeywords(task.task_description)) addRow(kw, task.id, 1.0);
    for (const kw of pyExtractKeywords(task.feature_area)) addRow(kw, task.id, 0.5);
  }
}

function buildStatus() {
  const generated = DB.raw.generated_at || '';
  DB.status = {
    role_count: DB.raw.role_count ?? DB.roles.length,
    task_count: DB.raw.task_count ?? DB.tasks.length,
    shadow_role_count: DB.raw.shadow_role_count ?? 0,
    pipeline: 'healthy',
    last_updated: generated ? String(generated).slice(0, 10) : null,
    generated_at: generated,
  };
}

// ===========================================================================
// Search — query tiers (port of runKeywordTier / runBM25Tier / runLikeTier)
// ===========================================================================

function inScope(task) {
  return task.out_of_scope == null;
}

// Build the SELECT-equivalent "row" for a matched task at a given base_score.
// Returns null when the min role is missing (mirrors the JOIN dropping the row).
function buildRow(taskId, baseScore, matchType) {
  const task = DB.tasksById.get(taskId);
  if (!task) return null;
  const role = DB.rolesById.get(task.min_role_id);
  if (!role) return null;
  return {
    row: {
      id: task.id,
      task_description: task.task_description,
      feature_area: task.feature_area,
      alt_role_ids: task.alt_role_ids,
      source_url: task.source_url,
      min_role_id: role.id,
      min_role_name: role.displayName,
      min_role_description: role.description || '',
      is_privileged: role.isPrivileged ? 1 : 0,
      permissions: role.permissions || [],
      base_score: baseScore,
    },
    matchType,
  };
}

// Phrase tier: lower(task_description) LIKE %q% — base_score 300, LIMIT 5
function phraseTier(q) {
  const qLower = q.toLowerCase();
  const out = [];
  for (const task of DB.tasks) {
    if (!inScope(task)) continue;
    if (task.task_description.toLowerCase().includes(qLower)) {
      out.push(buildRow(task.id, 300, 'exact'));
      if (out.length >= 5) break;
    }
  }
  return out.filter(Boolean);
}

// Full keyword tier (AND): tasks matching ALL keywords — base 20*distinctCount, LIMIT 5
function fullKeywordTier(kws) {
  const matchedKw = new Map(); // taskId -> Set(keyword)
  for (const kw of kws) {
    const rows = DB.kwIndex.get(kw);
    if (!rows) continue;
    for (const r of rows) {
      if (!matchedKw.has(r.taskId)) matchedKw.set(r.taskId, new Set());
      matchedKw.get(r.taskId).add(kw);
    }
  }
  const results = [];
  for (const [taskId, set] of matchedKw) {
    const task = DB.tasksById.get(taskId);
    if (!task || !inScope(task)) continue;
    if (set.size === kws.length) results.push({ taskId, score: set.size * 20 });
  }
  results.sort((a, b) => b.score - a.score);
  return results.slice(0, 5).map((x) => buildRow(x.taskId, x.score, 'full_keyword')).filter(Boolean);
}

// Partial keyword tier (OR): SUM(weight) over matching rows — LIMIT 10
function partialKeywordTier(kws) {
  const sum = new Map(); // taskId -> summed weight
  for (const kw of kws) {
    const rows = DB.kwIndex.get(kw);
    if (!rows) continue;
    for (const r of rows) sum.set(r.taskId, (sum.get(r.taskId) || 0) + r.weight);
  }
  const results = [];
  for (const [taskId, score] of sum) {
    const task = DB.tasksById.get(taskId);
    if (!task || !inScope(task)) continue;
    results.push({ taskId, score });
  }
  results.sort((a, b) => b.score - a.score);
  return results.slice(0, 10).map((x) => buildRow(x.taskId, x.score, 'partial')).filter(Boolean);
}

// Merge tiers into a deduped map keyed by task id (higher base_score wins).
function runKeywordTier(q, kws) {
  const merged = new Map();
  const add = (rows) => {
    for (const entry of rows) {
      if (!entry) continue;
      const id = entry.row.id;
      const score = entry.row.base_score || 0;
      const existing = merged.get(id);
      if (!existing || score > (existing.row.base_score || 0)) merged.set(id, entry);
    }
  };
  add(phraseTier(q));        // exact
  add(fullKeywordTier(kws)); // full_keyword
  add(partialKeywordTier(kws)); // partial
  return merged;
}

// BM25 tier (used by the debug=compare endpoint) — LIMIT 20
function runBM25Tier(kws) {
  if (kws.length === 0) return new Map();
  const avgdl = DB.corpus.avg_doc_length || 6.0;
  const sum = new Map();
  for (const kw of kws) {
    const rows = DB.kwIndex.get(kw);
    if (!rows) continue;
    for (const r of rows) {
      const task = DB.tasksById.get(r.taskId);
      const docLen = task && task.doc_length ? task.doc_length : avgdl;
      const term =
        r.idf * (r.tf * (BM25_K1 + 1)) /
        (r.tf + BM25_K1 * (1 - BM25_B + BM25_B * docLen / avgdl));
      sum.set(r.taskId, (sum.get(r.taskId) || 0) + term);
    }
  }
  const results = [];
  for (const [taskId, raw] of sum) {
    const task = DB.tasksById.get(taskId);
    if (!task || !inScope(task)) continue;
    results.push({ taskId, score: raw * 100 });
  }
  results.sort((a, b) => b.score - a.score);
  const map = new Map();
  for (const x of results.slice(0, 20)) {
    const entry = buildRow(x.taskId, x.score, 'bm25');
    if (entry) map.set(x.taskId, entry);
  }
  return map;
}

// LIKE fallback on the longest meaningful topic keyword (>=5 chars) — LIMIT 10
function runLikeTier(q, keywords, topicKeywords) {
  const meaningful = topicKeywords.filter((k) => k.length >= 5).sort((a, b) => b.length - a.length);
  const likeKw = (meaningful[0] || q).toLowerCase();
  const merged = new Map();
  for (const task of DB.tasks) {
    if (!inScope(task)) continue;
    if (
      task.task_description.toLowerCase().includes(likeKw) ||
      task.feature_area.toLowerCase().includes(likeKw)
    ) {
      const entry = buildRow(task.id, 1, 'partial');
      if (entry) merged.set(task.id, entry);
      if (merged.size >= 10) break;
    }
  }
  return merged;
}

// Apply privilege factor + affinity scoring + threshold + dedup (worker logic)
function applyAffinityAndScore(seen, srcQuery = '') {
  if (seen.size === 0) return [];

  const srcTokens = String(srcQuery)
    .toLowerCase()
    .split(/[\s\p{P}]+/u)
    .map((w) => w.trim())
    .filter((w) => w.length >= 2 && !STOP_WORDS.has(w));

  let scored = [...seen.values()].map(({ row, matchType }) => {
    const perms = row.permissions || [];
    const permCount = perms.length;
    const isPriv = row.is_privileged === 1;
    const baseScore = row.base_score || 0;
    const task = row.task_description;
    return {
      task,
      feature_area: row.feature_area,
      min_role: row.min_role_name,
      min_role_id: row.min_role_id,
      min_role_description: row.min_role_description || '',
      alt_roles: row.alt_role_ids || [],
      source_url: row.source_url,
      is_privileged: isPriv,
      permission_count: permCount,
      match_type: matchType,
      score: baseScore * privilegeFactor(permCount, isPriv),
      match_reasoning: generateMatchReasoning(srcQuery || task, task, matchType),
      relevant_permissions: rankRelevantPermissions(srcTokens, perms),
      _permCount: permCount,
    };
  });

  scored.sort((a, b) => (b.score !== a.score ? b.score - a.score : a._permCount - b._permCount));

  const dominantArea = scored[0].feature_area;
  for (const r of scored) r.score = r.score * affinityFactor(r.feature_area, dominantArea);

  scored.sort((a, b) => (b.score !== a.score ? b.score - a.score : a._permCount - b._permCount));

  const topScore = scored[0].score;
  const threshold = topScore * 0.05;

  const taskSeen = new Set();
  return scored
    .filter((r) => {
      if (r.score < threshold) return false;
      const key = String(r.task || '');
      if (taskSeen.has(key)) return false;
      taskSeen.add(key);
      return true;
    })
    .slice(0, 10)
    .map(({ _permCount, ...rest }) => rest);
}

// Resolve alt role ids -> { role_id, role_name, description }
function enrichAltRoles(results) {
  for (const r of results) {
    r.alt_roles_enriched = (r.alt_roles || [])
      .map((id) => {
        const role = DB.rolesById.get(id);
        return role
          ? { role_id: id, role_name: role.displayName, description: role.description || '' }
          : null;
      })
      .filter(Boolean);
  }
  return results;
}

function finalizeResults(seen, srcQuery) {
  const results = applyAffinityAndScore(seen, srcQuery);
  return enrichAltRoles(results);
}

function extractKeywordsForSearch(q) {
  const keywords = extractKeywords(q);
  const topicKeywords = keywords.filter((k) => !GENERIC_VERBS.has(k));
  const sqKeywords = topicKeywords.length > 0 ? topicKeywords : keywords;
  return { keywords, topicKeywords, sqKeywords };
}

function searchCompare(q) {
  const { keywords, topicKeywords, sqKeywords } = extractKeywordsForSearch(q);
  const keywordResults = applyAffinityAndScore(runKeywordTier(q, sqKeywords));
  const bm25Results = applyAffinityAndScore(runBM25Tier(sqKeywords));
  return {
    query: q,
    keywords,
    topic_keywords: topicKeywords,
    keyword_ranker: {
      count: keywordResults.length,
      top_5: keywordResults.slice(0, 5).map((r) => ({
        task: r.task, min_role: r.min_role, score: r.score, match_type: r.match_type,
      })),
    },
    bm25_ranker: {
      count: bm25Results.length,
      top_5: bm25Results.slice(0, 5).map((r) => ({
        task: r.task, min_role: r.min_role, score: r.score, match_type: r.match_type,
      })),
    },
    same_top_role:
      keywordResults[0]?.min_role != null &&
      keywordResults[0]?.min_role === bm25Results[0]?.min_role,
  };
}

function search(q, src) {
  if (!q) return { __badRequest: 'Missing or empty query parameter: q' };

  const { keywords, topicKeywords, sqKeywords } = extractKeywordsForSearch(q);
  if (keywords.length === 0) return [];

  const tier1 = runKeywordTier(q, sqKeywords);
  if (tier1.size > 0) return finalizeResults(tier1, src);

  if (topicKeywords.length < keywords.length) {
    const tier2 = runKeywordTier(q, keywords);
    if (tier2.size > 0) return finalizeResults(tier2, src);
  }

  const tier3 = runLikeTier(q, keywords, topicKeywords);
  if (tier3.size > 0) return finalizeResults(tier3, src);

  return [];
}

// ===========================================================================
// Routes
// ===========================================================================

// Permissive CORS (harmless; frontend is same-origin)
app.use((req, res, next) => {
  res.set('Access-Control-Allow-Origin', '*');
  next();
});

// GET /api/status
app.get('/api/status', (req, res) => {
  res.set('Cache-Control', 'no-store');
  res.json(DB.status);
});

// GET /api/roles  -> [{ id, display_name, is_privileged }]
app.get('/api/roles', (req, res) => {
  const roles = DB.roles
    .filter((r) => r.isBuiltIn)
    .sort((a, b) => String(a.displayName).localeCompare(String(b.displayName)))
    .map((r) => ({ id: r.id, display_name: r.displayName, is_privileged: !!r.isPrivileged }));
  res.json(roles);
});

// GET /api/search?q=&src=&debug=compare
app.get('/api/search', (req, res) => {
  const q = String(req.query.q || '').trim();
  const src = String(req.query.src || q).trim();
  if (!q) return res.status(400).json({ error: 'Missing or empty query parameter: q' });
  if (req.query.debug === 'compare') return res.json(searchCompare(q));

  const out = search(q, src);
  if (out && out.__badRequest) return res.status(400).json({ error: out.__badRequest });
  res.json(out);
});

// GET /api/diff?a=&b=
app.get('/api/diff', (req, res) => {
  const a = String(req.query.a || '').trim();
  const b = String(req.query.b || '').trim();
  if (!a || !b) return res.status(400).json({ error: 'Missing params: a and b (role display names)' });

  const roleA = DB.rolesByNameLower.get(a.toLowerCase());
  const roleB = DB.rolesByNameLower.get(b.toLowerCase());
  if (!roleA) return res.status(404).json({ error: `Role not found: ${a}` });
  if (!roleB) return res.status(404).json({ error: `Role not found: ${b}` });

  const permsA = new Set(roleA.permissions || []);
  const permsB = new Set(roleB.permissions || []);
  const onlyInA = [...permsA].filter((p) => !permsB.has(p)).sort();
  const onlyInB = [...permsB].filter((p) => !permsA.has(p)).sort();
  const shared = [...permsA].filter((p) => permsB.has(p)).sort();

  res.json({
    role_a: { id: roleA.id, display_name: roleA.displayName, permission_count: permsA.size, first_seen: null },
    role_b: { id: roleB.id, display_name: roleB.displayName, permission_count: permsB.size, first_seen: null },
    only_in_a: onlyInA,
    only_in_b: onlyInB,
    shared,
  });
});

// GET /api/quality  (no local Sentrux data -> has_data:false; frontend hides panel)
app.get('/api/quality', (req, res) => {
  res.json({ metrics: {}, has_data: false });
});

// GET /api/role/:id
app.get('/api/role/:id', (req, res) => {
  const role = DB.rolesById.get(req.params.id);
  if (!role) return res.status(404).json({ error: `Role not found: ${req.params.id}` });
  res.json({
    id: role.id,
    displayName: role.displayName,
    description: role.description,
    isPrivileged: !!role.isPrivileged,
    isBuiltIn: !!role.isBuiltIn,
    permissions: role.permissions || [],
    firstSeen: null,
    lastUpdated: DB.status.last_updated || null,
  });
});

// Serve the changelog + other data files locally (What's New panel)
app.use('/data', express.static(path.join(__dirname, 'data')));

// Static frontend
app.use(express.static(path.join(__dirname, 'frontend')));

// SPA fallback
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend', 'index.html'));
});

// ===========================================================================
// Boot
// ===========================================================================

try {
  loadData();
} catch (err) {
  console.error('FATAL: failed to load data/master.json:', err);
  process.exit(1);
}

app.listen(PORT, () => {
  console.log(`RoleLens server running on port ${PORT}`);
});
