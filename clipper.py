#!/usr/bin/env python3
"""Stream Clipper (free): YouTube link -> AI-selected vertical clips with captions.
Runs on GitHub Actions using GitHub Models as the free AI."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests
from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL

API_URL = "https://models.github.ai/inference/chat/completions"
CJK_LANGS = {"ja", "zh"}  # no spaces between words


# ----------------------------------------------------------------------------
# 1. Download
# ----------------------------------------------------------------------------
def download_video(url, work_dir, cookies=None):
    work_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": (
            "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
            "bestvideo[height<=1080]+bestaudio/best[height<=1080]"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(work_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "retries": 10,
        "extractor_retries": 3,
    }
    if cookies:
        opts["cookiefile"] = cookies
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    video_path = work_dir / f"{info['id']}.mp4"
    if not video_path.exists():
        raise RuntimeError(f"Download finished but {video_path} not found")
    return video_path, info


# ----------------------------------------------------------------------------
# 2. Transcribe (cached)
# ----------------------------------------------------------------------------
def get_transcript(video_path, cache, model_name, language, device):
    if cache.exists():
        print("  Using cached transcript.")
        return json.loads(cache.read_text(encoding="utf-8"))

    print(f"  Transcribing with whisper '{model_name}' (slow part)...", flush=True)
    compute = "int8" if device == "cpu" else "auto"
    model = WhisperModel(model_name, device=device, compute_type=compute)
    seg_iter, info = model.transcribe(
        str(video_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )
    segments, last = [], -5.0
    for seg in seg_iter:
        segments.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": [
                    {"start": w.start, "end": w.end, "word": w.word}
                    for w in (seg.words or [])
                ],
            }
        )
        pct = min(100.0, seg.end / info.duration * 100)
        if pct - last >= 5:
            print(f"  transcribing {pct:.0f}%", flush=True)
            last = pct

    data = {"language": info.language, "duration": info.duration, "segments": segments}
    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


# ----------------------------------------------------------------------------
# 3. Ask the free AI (GitHub Models) for the best moments
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert short-form video editor who cuts livestreams into \
YouTube Shorts that go viral. You read timestamped transcripts and pick the moments \
that work as standalone shorts.

A great short:
- Has a hook in the first 3 seconds (a bold claim, a question, a reaction, a funny setup).
- Is fully self-contained: a viewer who never saw the stream understands it.
- Has a clear payoff: a punchline, a huge reaction, a surprising story turn, a hot take, \
a useful insight, or real emotion.
- Starts at the beginning of a thought and ends right AFTER the payoff.
- Has no dead air and no long chat reading.

Timestamps are seconds from the start of the video. Use the exact timestamps shown. \
Be picky: returning fewer (or none) is better than returning weak clips.
You must reply with ONLY a JSON object, no other text."""


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def call_llm(model, user, retries=6):
    token = os.getenv("GITHUB_TOKEN")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/vnd.github+json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": 3000,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, headers=headers, json=body, timeout=180)
        except requests.RequestException as exc:
            print(f"  network error: {exc}")
            time.sleep(15)
            continue
        if r.status_code == 429:
            try:
                wait = int(float(r.headers.get("Retry-After", "")))
            except ValueError:
                wait = 30 * (attempt + 1)
            if wait > 300:
                print("  Rate limit reached (probably the daily free limit). Try again later.")
                return {"_limit": True}
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait + 1)
            continue
        if r.status_code != 200:
            print(f"  AI error {r.status_code}: {r.text[:300]}")
            return {}
        try:
            content = r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError):
            return {}
        time.sleep(4)  # stay under ~15 requests/minute
        return parse_json(content)
    return {}


def overlap(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def segment_text(segments, start, end):
    return " ".join(s["text"].strip() for s in segments if s["end"] > start and s["start"] < end)


def est_tokens(text):
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii + (len(text) - non_ascii) / 3.5


def build_windows(segs, budget=4500, overlap_n=5):
    """Split the transcript by token budget (free models have small input limits)."""
    windows, cur, used = [], [], 0.0
    for s in segs:
        text = s["text"].strip()
        if not text:
            continue
        item = (s["start"], f"[{s['start']:.0f}] {text}", est_tokens(text) + 4)
        if cur and used + item[2] > budget:
            windows.append(cur)
            cur = cur[-overlap_n:]
            used = sum(i[2] for i in cur)
        cur.append(item)
        used += item[2]
    if cur and (not windows or len(cur) > overlap_n):
        windows.append(cur)
    return windows


def dedupe(cands):
    keep = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        if all(
            overlap(c, k) < 0.4 * min(c["end"] - c["start"], k["end"] - k["start"])
            for k in keep
        ):
            keep.append(c)
    return keep


def find_clips(model, transcript, max_clips, min_len, max_len):
    segs = transcript["segments"]
    duration = transcript["duration"]
    windows = build_windows(segs)
    per_window = min(8, max(3, math.ceil(max_clips * 2.5 / max(1, len(windows)))))

    candidates = []
    for i, win in enumerate(windows, 1):
        print(f"  Scanning part {i}/{len(windows)} ({win[0][0]/60:.0f} min onward)", flush=True)
        user = (
            f"Find up to {per_window} moments in this transcript. Each clip must be between "
            f"{min_len} and {max_len} seconds long.\n"
            'Reply as JSON: {"clips":[{"start":number,"end":number,"title":"catchy title, '
            'max 60 chars, same language as transcript","reason":"one sentence",'
            '"score":number 1-10}]}\n\n' + "\n".join(x[1] for x in win)
        )
        result = call_llm(model, user)
        if result.get("_limit"):
            break
        for c in result.get("clips", []) if isinstance(result.get("clips"), list) else []:
            try:
                c = {
                    "start": float(c["start"]),
                    "end": float(c["end"]),
                    "title": str(c["title"]),
                    "reason": str(c["reason"]),
                    "score": float(c["score"]),
                }
            except (KeyError, ValueError, TypeError):
                continue
            length = c["end"] - c["start"]
            if c["start"] < 0 or c["end"] > duration + 1:
                continue
            if length < min_len - 8 or length > max_len + 20:
                continue
            candidates.append(c)

    candidates = dedupe(candidates)[:45]
    print(f"  {len(candidates)} candidates. Running final ranking pass...", flush=True)
    if not candidates:
        return []

    candidates.sort(key=lambda c: c["start"])
    for idx, c in enumerate(candidates):
        c["id"] = idx
    listing = "\n".join(
        f"ID {c['id']} | {c['start']:.0f}-{c['end']:.0f}s | score {c['score']:.0f} | "
        f"{c['title']} | {segment_text(segs, c['start'], c['end'])[:160]}"
        for c in candidates
    )
    user = (
        f"Here are {len(candidates)} candidate clips from one stream. Pick the best "
        f"{max_clips} (or fewer if the rest are weak). Prefer variety, strong hooks and clear "
        'payoffs. Reply as JSON: {"selected_ids":[ids, best first]}\n\n' + listing
    )
    picked = call_llm(model, user).get("selected_ids", [])
    if not isinstance(picked, list):
        picked = []
    picked = [i for i in picked if isinstance(i, int)]

    by_id = {c["id"]: c for c in candidates}
    ordered = [by_id[i] for i in picked if i in by_id]
    ordered += sorted((c for c in candidates if c["id"] not in picked), key=lambda c: -c["score"])

    final = []
    for c in ordered:
        if len(final) >= max_clips:
            break
        if all(overlap(c, f) < 0.5 for f in final):
            final.append(c)
    final.sort(key=lambda c: c["start"])
    return final


# ----------------------------------------------------------------------------
# 4. Snap cut points to word boundaries
# ----------------------------------------------------------------------------
def snap_to_words(transcript, clip, max_len):
    words = [w for s in transcript["segments"] for w in s["words"]]
    if not words:
        return clip["start"], clip["end"]
    start = min(words, key=lambda w: abs(w["start"] - clip["start"]))["start"]
    limit = start + max_len
    end_pool = [w for w in words if w["end"] <= limit] or words
    end = min(end_pool, key=lambda w: abs(w["end"] - clip["end"]))["end"]
    start = max(0.0, start - 0.2)
    end = min(transcript["duration"], end + 0.4)
    return start, end


# ----------------------------------------------------------------------------
# 5. Captions (ASS, word-by-word highlight)
# ----------------------------------------------------------------------------
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,7,2,2,60,60,420,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def build_ass(transcript, start, end, font):
    cjk = transcript["language"] in CJK_LANGS
    joiner = "" if cjk else " "
    max_chars = 9 if cjk else 20
    max_words = 99 if cjk else 4

    words = []
    for s in transcript["segments"]:
        for w in s["words"]:
            if w["start"] >= start - 0.05 and w["end"] <= end + 0.05:
                text = w["word"].strip().replace("{", "(").replace("}", ")")
                if text:
                    words.append({"s": w["start"] - start, "e": w["end"] - start, "t": text})

    groups, cur = [], []
    for w in words:
        cur.append(w)
        chars = len(joiner.join(x["t"] for x in cur))
        if chars >= max_chars or len(cur) >= max_words or re.search(r"[.!?。！？]$", w["t"]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)

    lines = [ASS_HEADER.format(font=font)]
    for g in groups:
        for i, w in enumerate(g):
            ev_start = w["s"]
            ev_end = g[i + 1]["s"] if i + 1 < len(g) else w["e"]
            if ev_end <= ev_start:
                ev_end = ev_start + 0.1
            parts = []
            for j, x in enumerate(g):
                if j == i:
                    parts.append("{\\c&H00FFFF&}" + x["t"] + "{\\c&HFFFFFF&}")  # yellow
                else:
                    parts.append(x["t"])
            lines.append(
                f"Dialogue: 0,{ass_time(ev_start)},{ass_time(ev_end)},Default,,0,0,0,,"
                + joiner.join(parts)
            )
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# 6. Render with ffmpeg
# ----------------------------------------------------------------------------
def render_clip(video_path, start, end, ass_text, out_dir, base_name, layout):
    ass_name = f"{base_name}.ass"
    (out_dir / ass_name).write_text(ass_text, encoding="utf-8")
    out_mp4 = out_dir / f"{base_name}.mp4"

    if layout == "crop":
        fc = f"[0:v]crop=ih*9/16:ih,scale=1080:1920,subtitles={ass_name}[v]"
    else:
        fc = (
            "[0:v]split=2[bg][fg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=30:4[b];"
            "[fg]scale=1080:-2[f];"
            f"[b][f]overlay=(W-w)/2:(H-h)/2,subtitles={ass_name}[v]"
        )

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
        "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video_path.resolve()),
        "-filter_complex", fc,
        "-map", "[v]", "-map", "0:a:0",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        str(out_mp4.resolve()),
    ]
    subprocess.run(cmd, cwd=out_dir, check=True)  # cwd so subtitles gets a plain filename
    return out_mp4


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def slugify(text, n=40):
    return re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")[:n] or "clip"


def process(url, args, model):
    print(f"\n=== {url}", flush=True)
    work_dir = Path("work")
    video_path, info = download_video(url, work_dir, args.cookies)
    print(f"  Downloaded: {info.get('title')}", flush=True)

    transcript = get_transcript(
        video_path, work_dir / f"{info['id']}.transcript.json",
        args.whisper_model, args.language, args.device,
    )

    print("  Finding the best moments...", flush=True)
    clips = find_clips(model, transcript, args.max_clips, args.min_len, args.max_len)
    if not clips:
        print("  No good clips found for this video.")
        return
    print(f"  Selected {len(clips)} clips. Rendering (slow preset, quality first)...", flush=True)

    out_dir = Path(args.out) / info["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for n, clip in enumerate(clips, 1):
        s, e = snap_to_words(transcript, clip, args.max_len)
        ass = build_ass(transcript, s, e, args.font)
        base = f"clip_{n:02d}_{slugify(clip['title'])}"
        print(f"  [{n}/{len(clips)}] {clip['title']}  ({s:.0f}s-{e:.0f}s)", flush=True)
        render_clip(video_path, s, e, ass, out_dir, base, args.layout)
        meta.append({"file": f"{base}.mp4", "title": clip["title"], "reason": clip["reason"],
                     "score": clip["score"], "start": round(s, 2), "end": round(e, 2),
                     "source": url})
    (out_dir / "clips.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Done. Clips are in: {out_dir}", flush=True)


def read_urls(args):
    urls = list(args.urls)
    env_urls = os.getenv("URLS", "")
    urls += [u for u in re.split(r"[\s,]+", env_urls) if u]
    if args.file:
        urls += [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    return urls


def main():
    p = argparse.ArgumentParser(description="Turn YouTube streams into captioned vertical clips.")
    p.add_argument("urls", nargs="*", help="YouTube links")
    p.add_argument("--file", help="text file with one link per line")
    p.add_argument("--max-clips", type=int, default=16)
    p.add_argument("--min-len", type=int, default=20)
    p.add_argument("--max-len", type=int, default=60)
    p.add_argument("--layout", choices=["blur", "crop"], default="blur")
    p.add_argument("--whisper-model", default="medium")
    p.add_argument("--language", default=None)
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    p.add_argument("--font", default="Noto Sans CJK JP")
    p.add_argument("--cookies", default=None, help="Netscape cookies.txt for YouTube")
    p.add_argument("--out", default="output")
    args = p.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH.")
    if not os.getenv("GITHUB_TOKEN"):
        sys.exit("GITHUB_TOKEN missing (it is provided automatically in GitHub Actions).")

    urls = read_urls(args)
    if not urls:
        sys.exit("No links given.")

    model = os.getenv("CLIP_MODEL", "openai/gpt-4.1-mini")
    failed = 0
    for url in urls:
        try:
            process(url, args, model)
        except Exception as exc:
            failed += 1
            print(f"  FAILED on {url}: {exc}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
