// Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, minute by minute, 4 hours
// ahead, as five quantiles a minute (Node 22, standard library only).
//
// Approach: persistence median plus an empirical, adaptive spread.
//
// The prices barely move most minutes, then jump: a normal random walk is the
// wrong shape (its 25%/75% bands are far too wide at every quiet minute, while
// its tails miss the jumps). So the spread comes from the case's own recent
// history, not from a fixed sigma:
//
//   1. Pooled residual table. From the last 8 hours of every market with enough
//      history, pool the h-step price changes at horizons 1, 5, 15, 30, 60, 120
//      and 240 minutes and take their 5/25/50/75/95% quantiles. This captures
//      the true shape: a spike at zero, tight inner quantiles, fat asymmetric
//      tails, and sub-square-root growth with horizon (prices mean-revert).
//   2. Per-market activity. Each market's band is scaled by its recent
//      restlessness (mean absolute 1-minute move over the last 4 hours,
//      relative to the cross-market mean, shrunk 25% toward 1 and capped at
//      2.5x). Dead markets get narrow bands, jumpy ones wide ones, and the
//      scale self-calibrates as the regime changes.
//   3. The median is the last price. A pull toward a trailing average looked
//      strong in-sample but lost out-of-sample (it fights trends), so the
//      forecast keeps the martingale median and puts all the structure in the
//      spread.
//
// Validation (local scorer; ground truth for p00-p04 taken from the next
// case's file, overlap verified byte-exact on 24,960 minutes): loss 0.317424
// on p00-p04 vs the house baseline's 0.392518 on the same cases (-19%),
// 90% coverage 0.934. The design was chosen on an out-of-sample split (tables
// fit only on prices before 2026-11-12T12:00Z, scored on p02-p04): pooled
// table + activity scaling 0.286, vs baseline 0.355; per-market tables and a
// mean-reversion pull both failed that split and were dropped. A normal
// random-walk spread with per-market sigma scored 0.468+ and was rejected:
// the peaked, jumpy return shape needs empirical quantiles, not z-scores.
//
// Everything is a deterministic function of the case file: two runs write the
// same bytes. No network, no randomness, no clock.
//
// Input:  every <id>.in in $MAKEFUN_INPUT_DIR (else /input): one case, the
//         markets' 1-minute Yes prices up to the checkpoint.
// Output: <id>.out in $MAKEFUN_OUTPUT_DIR (else /output):
//         {"forecasts":[{"id","q"}]}, q exactly horizon_minutes rows of the
//         five quantiles in level order, each finite in [0,1], non-decreasing.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Horizons (minutes) at which the pooled residual table is tabulated. */
const ANCHORS = [1, 5, 15, 30, 60, 120, 240];
/** Minutes of recent history each market contributes to the pooled table. */
const FIT_WINDOW = 480;
/** A market needs at least this many minutes in the fit window to join the pool. */
const MIN_FIT_POINTS = 300;
/** Minutes of recent history behind each market's activity scale. */
const ACTIVITY_WINDOW = 240;
/** Activity is shrunk this far from 1: act = 1 + SHRINK * (raw - 1). */
const ACTIVITY_SHRINK = 0.75;
/** Activity is clamped to [1 / ACTIVITY_CAP, ACTIVITY_CAP]. */
const ACTIVITY_CAP = 2.5;
/** Fallback spread (house baseline) if a case holds too little history to fit. */
const FALLBACK_SIGMA = 0.0025;
const NORMAL_Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Linear-interpolation quantile of a sorted array. */
function quantileOf(sorted: number[], q: number): number {
  const idx = (sorted.length - 1) * q;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return sorted[lo] as number + ((sorted[hi] as number) - (sorted[lo] as number)) * (idx - lo);
}

/**
 * Fit the pooled residual table: for each anchor horizon, the quantiles of the
 * h-step price changes pooled over the given series. A series shorter than the
 * horizon contributes nothing to that anchor. Empty anchors inherit the nearest
 * non-empty anchor's quantiles, scaled by sqrt(h / hRef), so the table is total.
 */
function fitPooledTable(seriesList: number[][]): number[][] {
  const buckets: number[][] = ANCHORS.map(() => []);
  for (const p of seriesList) {
    for (let a = 0; a < ANCHORS.length; a++) {
      const h = ANCHORS[a] as number;
      for (let i = 0; i + h < p.length; i++) {
        (buckets[a] as number[]).push((p[i + h] as number) - (p[i] as number));
      }
    }
  }
  const table: number[][] = [];
  let lastFilled = -1;
  for (let a = 0; a < ANCHORS.length; a++) {
    const b = buckets[a] as number[];
    if (b.length > 0) {
      b.sort((x, y) => x - y);
      table.push(QUANTILES.map((q) => quantileOf(b, q)));
      lastFilled = a;
    } else if (lastFilled >= 0) {
      const scale = Math.sqrt((ANCHORS[a] as number) / (ANCHORS[lastFilled] as number));
      table.push((table[lastFilled] as number[]).map((v) => v * scale));
    } else {
      table.push([]); // filled from the next non-empty anchor below
    }
  }
  // Any leading empty anchors take the first non-empty anchor, scaled down.
  const firstFilled = table.findIndex((row) => row.length > 0);
  if (firstFilled < 0) return [];
  for (let a = 0; a < firstFilled; a++) {
    const scale = Math.sqrt((ANCHORS[a] as number) / (ANCHORS[firstFilled] as number));
    table[a] = (table[firstFilled] as number[]).map((v) => v * scale);
  }
  return table;
}

/** The table's quantile at horizon h (linear in h between anchors), level j. */
function tableAt(table: number[][], h: number, j: number): number {
  if (h <= (ANCHORS[0] as number)) return (table[0] as number[])[j] as number;
  for (let a = 1; a < ANCHORS.length; a++) {
    if (h <= (ANCHORS[a] as number)) {
      const t = (h - (ANCHORS[a - 1] as number)) / ((ANCHORS[a] as number) - (ANCHORS[a - 1] as number));
      const lo = (table[a - 1] as number[])[j] as number;
      const hi = (table[a] as number[])[j] as number;
      return lo + (hi - lo) * t;
    }
  }
  return (table[table.length - 1] as number[])[j] as number;
}

/** Mean absolute 1-minute move over the last `win` prices (0 when too short). */
function meanAbsMove(prices: number[], win: number): number {
  const p = prices.slice(-win);
  if (p.length < 2) return 0;
  let s = 0;
  for (let i = 1; i < p.length; i++) s += Math.abs((p[i] as number) - (p[i - 1] as number));
  return s / (p.length - 1);
}

/** Forecast one case: persistence median, empirical adaptive spread. */
function forecastCase(c: Case): { id: string; q: number[][] }[] {
  // Pooled table from the recent history of markets with enough of it.
  const pool: number[][] = [];
  for (const m of c.markets) {
    const slice = m.prices.slice(-FIT_WINDOW);
    if (slice.length >= MIN_FIT_POINTS) pool.push(slice);
  }
  const table = fitPooledTable(pool);

  // Per-market activity vs the cross-market mean (1 when nothing moves).
  const moves = c.markets.map((m) => meanAbsMove(m.prices, ACTIVITY_WINDOW));
  const ref = moves.reduce((a, b) => a + b, 0) / Math.max(1, moves.length);

  return c.markets.map((m, mi) => {
    if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
    const x = m.prices[m.prices.length - 1] as number;
    let act = 1;
    if (ref > 0) {
      const raw = (moves[mi] as number) / ref;
      act = Math.min(ACTIVITY_CAP, Math.max(1 / ACTIVITY_CAP, 1 + ACTIVITY_SHRINK * (raw - 1)));
    }
    const rows: number[][] = [];
    for (let h = 1; h <= c.horizon_minutes; h++) {
      let row: number[];
      if (table.length > 0) {
        row = QUANTILES.map((_, j) => price6(x + act * tableAt(table, h, j)));
      } else {
        // Degenerate case (no fittable history anywhere): house baseline shape.
        row = NORMAL_Z.map((z) => price6(x + z * FALLBACK_SIGMA * Math.sqrt(h)));
      }
      for (let j = 1; j < row.length; j++) {
        if ((row[j] as number) < (row[j - 1] as number)) row[j] = row[j - 1] as number;
      }
      rows.push(row);
    }
    return { id: m.id, q: rows };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) {
    throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  }
  const forecasts = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
