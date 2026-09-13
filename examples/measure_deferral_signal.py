"""Measure whether a cheap model's token-level confidence is a usable deferral signal.

Study 3 proposes a cascade: a cheap model answers every candidate pair, and only the
pairs it is unsure about escalate to the expensive matcher. That design is worth
building only if the cheap model's confidence actually separates its right answers from
its wrong ones. An earlier attempt to predict escalation from hand-crafted string
features reached AUC 0.660 and lost to the expensive model alone at every budget, so the
question here is whether asking the model itself does better.

The expensive tier's answers already exist in the run being replayed, so the whole
cascade can be simulated without a single extra deepseek call: only the cheap tier is
queried. deepseek-v4-flash-0731 returns no logprobs, but the cheap tier does not have to
be the same model -- gpt-4o-mini exposes them, which is what makes this measurable.

Usage:
    python -m examples.measure_deferral_signal --dataset DBLP-ACM --limit 13080
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from code.llm_client import get_llm_client
from code.matcher import PROMPT, clean_record

CHEAP_MODEL = "openai/gpt-4o-mini"


def p_yes(record_a, record_b, client, model):
    """Return (answer, P(Yes)) from the cheap model's first-token distribution."""
    prompt = (PROMPT
              .replace("{record_a}", json.dumps(record_a, ensure_ascii=False))
              .replace("{record_b}", json.dumps(record_b, ensure_ascii=False)))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=4,
        logprobs=True,
        top_logprobs=8,
    )
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    answer = "Yes" if text.lower().startswith("yes") else "No"

    # The answer is one token, so the first position's top-k distribution is the whole
    # decision. Yes-mass and No-mass are summed separately because tokenizers split the
    # two labels across several surface forms (" Yes", "yes", "YES").
    yes_mass = no_mass = 0.0
    if choice.logprobs and choice.logprobs.content:
        for alt in choice.logprobs.content[0].top_logprobs:
            tok = alt.token.strip().lower()
            if tok.startswith("yes"):
                yes_mass += float(np.exp(alt.logprob))
            elif tok.startswith("no"):
                no_mass += float(np.exp(alt.logprob))
    total = yes_mass + no_mass
    # A distribution carrying neither label says nothing about the decision; 0.5 marks it
    # maximally uncertain so it escalates first rather than being trusted.
    return answer, (yes_mass / total if total > 0 else 0.5)


def auc(scores, labels):
    """Rank-based AUC; ties share their average rank."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = np.asarray(scores)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    pos = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def f1(pred, gold):
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="run directory whose final_results.csv supplies the pairs, the "
                         "per-pair attribute subsets, and the expensive tier's answers")
    ap.add_argument("--table-a", required=True)
    ap.add_argument("--table-b", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--gt-cols", nargs=2, required=True)
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=CHEAP_MODEL)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    a_raw = pd.read_csv(args.table_a, encoding=args.encoding)
    b_raw = pd.read_csv(args.table_b, encoding=args.encoding)
    ids_a = [str(v).strip() for v in a_raw["id"]]
    ids_b = [str(v).strip() for v in b_raw["id"]]
    df_a = a_raw.drop(columns=["id"], errors="ignore")
    df_b = b_raw.drop(columns=["id"], errors="ignore")

    gt = pd.read_csv(args.gold, encoding=args.encoding, sep=args.sep)
    ia = {v: i for i, v in enumerate(ids_a)}
    ib = {v: i for i, v in enumerate(ids_b)}
    ca, cb = args.gt_cols
    gold = {(ia[str(r[ca]).strip()], ib[str(r[cb]).strip()])
            for _, r in gt.iterrows()
            if str(r[ca]).strip() in ia and str(r[cb]).strip() in ib}

    res = pd.read_csv(os.path.join(args.run_dir, "final_results.csv"))
    if args.limit:
        res = res.head(args.limit)
    print(f"pairs: {len(res)}   gold in candidate set: "
          f"{sum(1 for x, y in zip(res.indexA, res.indexB) if (x, y) in gold)}"
          f"   total gold: {len(gold)}", flush=True)

    client = get_llm_client()
    rows = [None] * len(res)

    # Answers are appended as they arrive and reloaded on restart. Without this a run
    # killed at 57% -- which is what happened -- discards every call it had paid for.
    ckpt_path = (args.out or "deferral") + ".partial.jsonl"
    done_pos = set()
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn final line from a kill mid-write
                if rec["pos"] < len(rows):
                    rows[rec["pos"]] = (rec["answer"], rec["p"])
                    done_pos.add(rec["pos"])
        print(f"resuming: {len(done_pos)} pairs already answered", flush=True)
    ckpt = open(ckpt_path, "a", buffering=1)
    ckpt_lock = __import__("threading").Lock()

    def work(pos, ia_, ib_, attrs):
        ra = {k: v for k, v in df_a.iloc[ia_].to_dict().items() if k in attrs}
        rb = {k: v for k, v in df_b.iloc[ib_].to_dict().items() if k in attrs}
        ans, p = p_yes(clean_record(ra), clean_record(rb), client, args.model)
        return pos, ans, p

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for pos, (ia_, ib_, attrs) in enumerate(
                zip(res.indexA, res.indexB, res.selected_attributes)):
            if pos in done_pos:
                continue
            futs[ex.submit(work, pos, ia_, ib_, json.loads(attrs))] = pos
        done = 0
        for fut in as_completed(futs):
            try:
                pos, ans, p = fut.result()
                rows[pos] = (ans, p)
                with ckpt_lock:
                    ckpt.write(json.dumps({"pos": pos, "answer": ans, "p": p}) + "\n")
            except Exception as e:  # a failed cheap call must escalate, not guess
                rows[futs[fut]] = ("Error", 0.5)
                if done < 5:
                    print(f"  call failed: {e}", file=sys.stderr, flush=True)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(res)}", flush=True)

    ckpt.close()
    rows = [r if r is not None else ("Error", 0.5) for r in rows]
    res["cheap_answer"] = [r[0] for r in rows]
    res["cheap_p_yes"] = [r[1] for r in rows]
    res["gold"] = [(x, y) in gold for x, y in zip(res.indexA, res.indexB)]
    if args.out:
        res.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")

    gold_arr = res.gold.values
    cheap = (res.cheap_answer == "Yes").values
    exp = (res.answer == "Yes").values
    # Confidence in the answer given, not in "Yes": the deferral question is how sure the
    # model is of whatever it just said.
    conf = np.where(cheap, res.cheap_p_yes, 1 - res.cheap_p_yes)
    correct = cheap == gold_arr

    n_missing = len(gold) - int(gold_arr.sum())
    print("\n=== tier accuracy (end-to-end, blocking misses counted as FN) ===")
    for tag, pred in [(f"cheap  {args.model}", cheap), ("expensive (from run)", exp)]:
        tp = int((pred & gold_arr).sum()); fp = int((pred & ~gold_arr).sum())
        fn = int((~pred & gold_arr).sum()) + n_missing
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        print(f"  {tag:24s} F1={2*p*r/(p+r) if p+r else 0:.4f} P={p:.4f} R={r:.4f}")
    print(f"  the two tiers disagree on {int((cheap != exp).sum())} pairs "
          f"({(cheap != exp).mean()*100:.1f}%)")

    print("\n=== is confidence a usable deferral signal? ===")
    print(f"  AUC(confidence -> cheap answer correct) = {auc(conf, correct):.4f}")
    print("  (0.5 = useless, >0.8 = strong)")

    print("\n=== simulated cascade: escalate the least-confident pairs ===")
    print("  escalate%   F1      vs cheap-only   vs expensive-only")
    order = np.argsort(conf)
    base_cheap = f1(cheap, gold_arr)
    base_exp = f1(exp, gold_arr)
    for frac in [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]:
        k = int(len(res) * frac)
        pred = cheap.copy()
        pred[order[:k]] = exp[order[:k]]
        v = f1(pred, gold_arr)
        print(f"  {frac*100:6.0f}%   {v:.4f}     {v-base_cheap:+.4f}        {v-base_exp:+.4f}")
    print(f"\n  cheap-only F1={base_cheap:.4f}   expensive-only F1={base_exp:.4f}")
    print("  A cascade is worth building only if some row beats expensive-only, or ties "
          "it at a small escalation fraction.")


if __name__ == "__main__":
    main()
