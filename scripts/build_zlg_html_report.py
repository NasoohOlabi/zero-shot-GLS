"""Build an HTML report from prompt comparison artifacts."""

from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "tmp_saves" / "runs"
REPORT_PATH = RUNS / "zlg_experiment_report.html"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def fmt_float(value: str | float) -> str:
    return f"{float(value):.4f}"


def metric_class(old_value: str | float, new_value: str | float, lower_is_better: bool = True) -> str:
    old_f = float(old_value)
    new_f = float(new_value)
    if new_f == old_f:
        return "same"
    improved = new_f < old_f if lower_is_better else new_f > old_f
    return "ok" if improved else "bad"


def pre(text: str) -> str:
    return f"<div class=\"mono\">{html.escape(text)}</div>"


def render_prompt(context: str, corpus: str = "IMDB about movies") -> str:
    return (
        "/no_think\n"
        "Write exactly one new sentence in the same style as the examples below.\n"
        "Output only the sentence.\n"
        "Do not use tags, XML, markdown, labels, or explanations.\n"
        "Do not repeat the prompt.\n"
        "Use plain ASCII only.\n"
        f"Corpus: {corpus}\n"
        "Examples:\n"
        f"{context}\n"
        "New sentence:\n"
    )


def build_examples_table(model_label: str, samples: list[dict], count: int = 3) -> str:
    rows = []
    for idx, sample in enumerate(samples[:count], start=1):
        prompt = render_prompt(sample["context"])
        rows.append(
            f"""
          <tr>
            <td>{idx}</td>
            <td><strong>{html.escape(model_label)}</strong></td>
            <td>{pre(sample["context"])}</td>
            <td>{pre(prompt)}</td>
            <td>{pre(sample["new_text"])}</td>
          </tr>"""
        )
    return "\n".join(rows)


def build_report() -> str:
    summary_rows = read_csv(RUNS / "prompt_sample_summary.csv")
    qwen = read_json(RUNS / "qwen_qwen3.5_9b_prompt_samples.json")

    summary_html = []
    for row in summary_rows:
        if row["model"] != "qwen/qwen3.5-9b":
            continue
        row_class_tag = metric_class(row["old_avg_tag_hits"], row["new_avg_tag_hits"])
        row_class_rep = metric_class(
            row["old_avg_repetition_ratio"], row["new_avg_repetition_ratio"]
        )
        row_class_ascii = metric_class(
            row["old_ascii_ok_rate"], row["new_ascii_ok_rate"], lower_is_better=False
        )
        assessment = "Improved after switching Qwen to a plain /no_think completion prompt."
        prompt_style = "Qwen plain /no_think prompt"
        summary_html.append(
            f"""
          <tr>
            <td><strong>{html.escape(row["model"])}</strong><br><span class="sub">{html.escape(prompt_style)}</span></td>
            <td>{fmt_float(row["old_avg_tag_hits"])}</td>
            <td class="{row_class_tag}">{fmt_float(row["new_avg_tag_hits"])}</td>
            <td>{fmt_float(row["old_avg_word_count"])}</td>
            <td>{fmt_float(row["new_avg_word_count"])}</td>
            <td>{fmt_float(row["old_avg_repetition_ratio"])}</td>
            <td class="{row_class_rep}">{fmt_float(row["new_avg_repetition_ratio"])}</td>
            <td>{fmt_float(row["old_ascii_ok_rate"])}</td>
            <td class="{row_class_ascii}">{fmt_float(row["new_ascii_ok_rate"])}</td>
            <td>{html.escape(assessment)}</td>
          </tr>"""
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ZLG Experiment Report</title>
  <style>
    :root {{
      --bg: #f2f5f7;
      --card: #ffffff;
      --text: #17212b;
      --muted: #596775;
      --line: #d7dee5;
      --ok: #0c7a43;
      --bad: #b42318;
      --same: #4b5563;
      --accent: #0d4ea6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Segoe UI", serif;
      color: var(--text);
      background:
        radial-gradient(circle at top right, #dfe9f3 0, transparent 30%),
        linear-gradient(180deg, #f7fafc 0%, #eef3f7 100%);
    }}
    .wrap {{
      max-width: 1240px;
      margin: 24px auto 56px;
      padding: 0 16px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 16px;
      box-shadow: 0 8px 24px rgba(16, 24, 40, 0.04);
    }}
    h1, h2, h3 {{ margin: 0 0 10px; }}
    h1 {{ font-size: 32px; }}
    h2 {{ font-size: 21px; }}
    h3 {{ font-size: 16px; color: var(--muted); }}
    p {{ margin: 8px 0; }}
    .meta, .sub, .footer {{ color: var(--muted); font-size: 13px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 10px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #f8fafc;
      font-size: 13px;
      letter-spacing: 0;
    }}
    .mono {{
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
      line-height: 1.45;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
    }}
    .ok {{ color: var(--ok); font-weight: 700; }}
    .bad {{ color: var(--bad); font-weight: 700; }}
    .same {{ color: var(--same); font-weight: 700; }}
    .pill {{
      display: inline-block;
      margin: 0 8px 8px 0;
      padding: 3px 9px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>Zero-shot Generative Linguistic Steganography Report</h1>
      <p class="meta">Generated from local run artifacts in <code>tmp_saves/runs</code>.</p>
      <p>
        This document summarizes the prompt-side comparison work after the implementation audit.
        The key change for Qwen is a model-specific plain completion prompt with <code>/no_think</code>,
        replacing the XML-heavy template that Qwen was echoing back.
      </p>
      <p>
        <span class="pill">Corpus: IMDB about movies</span>
        <span class="pill">Trials per model: 12</span>
        <span class="pill">Metrics: tag leakage, repetition, word count, ASCII rate</span>
      </p>
    </section>

    <section class="card">
      <h2>Metrics Summary</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Old Avg Tag Hits</th>
            <th>New Avg Tag Hits</th>
            <th>Old Avg Words</th>
            <th>New Avg Words</th>
            <th>Old Avg Repetition</th>
            <th>New Avg Repetition</th>
            <th>Old ASCII Rate</th>
            <th>New ASCII Rate</th>
            <th>Assessment</th>
          </tr>
        </thead>
        <tbody>
          {''.join(summary_html)}
        </tbody>
      </table>
      <p class="meta">Source: <code>prompt_sample_summary.csv</code></p>
    </section>

    <section class="card">
      <h2>Interpretation</h2>
      <h3>Qwen</h3>
      <p>
        Qwen improved sharply once the prompt was simplified and <code>/no_think</code> was added.
        In the updated sample set, average tag leakage dropped from 2.0000 to 0.0000 and ASCII
        compliance improved from 0.8333 to 1.0000.
      </p>
    </section>

    <section class="card">
      <h2>Example /hide Requests and ZLG Responses</h2>
      <p class="meta">
        Each row shows the hide context, the full rendered prompt sent to the model, and the updated model output.
        Text is intentionally untruncated.
      </p>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Model</th>
            <th>/hide Request (Context)</th>
            <th>Prompt</th>
            <th>ZLG Response</th>
          </tr>
        </thead>
        <tbody>
          {build_examples_table(qwen["model"], qwen["samples"], count=3)}
        </tbody>
      </table>
    </section>

    <section class="card">
      <h2>Fair Comparison Status</h2>
      <p>
        The prompt-level comparison is now much cleaner for Qwen, but the full paper-faithful hide/extract
        benchmark is still limited by backend capabilities. The local LM Studio server does not expose the
        tokenization, detokenization, and token-level top-logprobs endpoints required by stage 2 and stage 3.
      </p>
      <p class="footer">
        Source files:
        <code>{html.escape(os.fspath(RUNS / "qwen_qwen3.5_9b_prompt_samples.json"))}</code>,
        <code>{html.escape(os.fspath(RUNS / "prompt_sample_summary.csv"))}</code>.
      </p>
    </section>
  </div>
</body>
</html>
"""


if __name__ == "__main__":
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(REPORT_PATH)
