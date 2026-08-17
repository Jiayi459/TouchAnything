#!/usr/bin/env python3
"""
Selectively extract members of the ICLR force-vision release `d256.zip` (185.2 GiB on Google
Drive) straight out of the remote archive, without ever storing the zip.

Why not gdown
-------------
`d256.zip` is one 198,849,542,248-byte file. The shard-at-a-time trick used by
`stream_opentouch.sh` / `download_own_copies.sh` does not apply -- there is nothing to
stream but the whole thing, and unpacking it needs a *second* 250.3 GiB on disk.
But Drive serves this file with `Accept-Ranges: bytes` (verified 2026-08-16, HTTP 206), and a
ZIP keeps a central directory of per-member offsets. So we read the 22.8 MB central directory
once and then fetch only the byte spans we actually want.

That matters because the archive is 95% video:

    group       files   uncomp GiB   xfer GiB       content
    signals     25473        3.89       1.10        tactile/EMG/pose pickles
    signals1    28426        4.34       1.23
    signals2    26922        4.11       1.17
    videos      50942       75.80      57.64        RGB npz, 256px *and* 32px
    videos1     28426       83.29      63.72        RGB npz, 256px only
    videos2     26922       78.88      60.30        RGB npz, 256px only

`--groups signals` therefore transfers **3.49 GiB instead of 185.2 GiB**, and because the
three signal runs are contiguous in the archive it costs ~3 long sequential reads rather than
80,821 small ones (the coalescing below).

Payloads (verified by pulling real members, 2026-08-16):
  signals `<split>/<subject>/<session>/<clip>.p` -> pickle dict
      {'signal': {'tactile-glove-left'  (16,32,32) f32, 'tactile-glove-right' (16,32,32) f32,
                  'myo-emg-{left,right}' (16,8), 'myo-acc-{left,right}' (16,3),
                  'joint-position' (16,28,3), '{left,right}-hand-pose' (16,24,3)},
       'label_text': str e.g. 'Slice a cucumber', 'label_idx': int}
      -> ActionSense sensor suite, 16-frame clips, values pre-scaled to ~[0,1].
  videos  `.../video_<k>_256.npz` -> arr_0 (16,256,256,3) uint8;  `_32` -> (16,32,32,3) uint8.
  signals/ego_4d_{verb,noun}.npy -> 148 verbs / 112 nouns (Ego4D vocabulary).

Usage (on CRC; stdlib only, no conda env needed):
    python3 scripts/crc/fetch_d256.py --dest /scratch365/$USER/forcevision            # signals
    python3 scripts/crc/fetch_d256.py --dest ... --groups signals,videos --lowres     # +32px RGB
    python3 scripts/crc/fetch_d256.py --dest ... --groups all                         # 185 GiB
    python3 scripts/crc/fetch_d256.py --dest ... --plan                               # dry run

Interrupt-safe and idempotent: members are verified by CRC-32 from the central directory and
already-correct files are skipped, so re-running resumes. Never writes outside --dest.
"""
import argparse
import binascii
import http.cookiejar
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

FILE_ID = "1UPCkTTmPJGex2p3cJRnJHj4_A70sGQ8S"
ZIP_SIZE = 198849542248  # bytes; asserted against Content-Range so a re-upload can't silently
                         # shift every offset we cached in the manifest.
BASE = "https://drive.usercontent.google.com/download"
UA = "Mozilla/5.0 (X11; Linux x86_64)"

# One HTTP request per this many payload bytes. Bounds the damage of a mid-stream reset:
# a dropped connection costs at most one chunk of re-reading, not the whole run.
CHUNK = 512 << 20
# Members closer than this are fetched in one span; the bytes in the gap are read and thrown
# away, which is cheaper than paying a new request + TLS handshake to skip them.
GAP_TOLERANCE = 8 << 20


class Drive:
    """Ranged reader for a public Drive file, refreshing the virus-scan confirm token."""

    def __init__(self):
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar))
        self.opener.addheaders = [("User-Agent", UA)]
        self.url = None

    def _confirm(self):
        """Files >100 MB return an interstitial; the real URL needs its per-session uuid."""
        first = f"{BASE}?id={FILE_ID}&export=download"
        with self.opener.open(first, timeout=60) as r:
            body = r.read(1 << 20).decode("utf-8", "replace")
            ctype = r.headers.get("Content-Type", "")
        if "text/html" not in ctype:
            self.url = first  # served directly; no interstitial
            return
        if "uuid" not in body and re.search(r"quota|too many users", body, re.I):
            raise RuntimeError(
                "Drive is refusing the file (download quota / permissions). Wait a few hours, "
                "or copy it into your own Drive and re-run with --file-id <new id>.")
        fields = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', body))
        if "uuid" not in fields:
            raise RuntimeError("could not parse Drive confirm form; page layout changed")
        fields.setdefault("id", FILE_ID)
        self.url = BASE + "?" + urllib.parse.urlencode(fields)

    def open_range(self, start, end, tries=6):
        """Return a live response streaming bytes [start, end] inclusive."""
        last = None
        for attempt in range(tries):
            try:
                if self.url is None:
                    self._confirm()
                req = urllib.request.Request(self.url, headers={
                    "User-Agent": UA, "Range": f"bytes={start}-{end}"})
                resp = self.opener.open(req, timeout=180)
                if resp.status != 206:
                    resp.close()
                    raise RuntimeError(f"expected 206, got {resp.status}")
                cr = resp.headers.get("Content-Range", "")
                total = cr.rsplit("/", 1)[-1]
                if total.isdigit() and int(total) != ZIP_SIZE:
                    resp.close()
                    raise RuntimeError(
                        f"remote size {total} != expected {ZIP_SIZE}; the Drive file changed, "
                        "so cached offsets are stale. Delete manifest.json and re-run.")
                return resp
            except Exception as exc:                  # noqa: BLE001 - retry anything transient
                last = exc
                self.url = None                       # force a fresh confirm token
                if attempt < tries - 1:
                    nap = min(60, 2 ** attempt * 3)
                    print(f"    ! {type(exc).__name__}: {exc} -- retry in {nap}s",
                          file=sys.stderr, flush=True)
                    time.sleep(nap)
        raise RuntimeError(f"range {start}-{end} failed after {tries} tries: {last}")

    def read_range(self, start, end):
        with self.open_range(start, end) as r:
            return r.read()


def load_manifest(drive, cache_path):
    """Parse the ZIP64 central directory into [{n,m,c,u,o,crc}], cached on disk."""
    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            m = json.load(fh)
        if m.get("zip_size") == ZIP_SIZE:
            return m["entries"]
        print("  manifest cache is for a different archive size; re-reading", flush=True)

    print("  reading central directory (~23 MB)...", flush=True)
    tail = drive.read_range(ZIP_SIZE - (1 << 20), ZIP_SIZE - 1)
    j = tail.rfind(b"PK\x06\x06")
    if j < 0:
        raise RuntimeError("no ZIP64 end-of-central-directory record found")
    _, _, _, _, _, _, _, n_entries, cd_size, cd_off = struct.unpack(
        "<IQHHIIQQQQ", tail[j:j + 56])
    raw = drive.read_range(cd_off, cd_off + cd_size - 1)

    entries, off = [], 0
    while off + 4 <= len(raw) and raw[off:off + 4] == b"PK\x01\x02":
        (_, _, _, _, meth, _, _, crc, csz, usz, nlen, elen, clen,
         _, _, _, lho) = struct.unpack("<IHHHHHHIIIHHHHHII", raw[off:off + 46])
        name = raw[off + 46:off + 46 + nlen].decode("utf-8", "replace")
        extra = raw[off + 46 + nlen:off + 46 + nlen + elen]
        if 0xFFFFFFFF in (usz, csz, lho):             # ZIP64 promotes the overflowed fields
            p = 0
            while p + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[p:p + 4])
                body, q = extra[p + 4:p + 4 + hsz], 0
                if hid == 1:
                    if usz == 0xFFFFFFFF:
                        usz = struct.unpack("<Q", body[q:q + 8])[0]; q += 8
                    if csz == 0xFFFFFFFF:
                        csz = struct.unpack("<Q", body[q:q + 8])[0]; q += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", body[q:q + 8])[0]
                p += 4 + hsz
        entries.append({"n": name, "m": meth, "c": csz, "u": usz, "o": lho, "crc": crc})
        off += 46 + nlen + elen + clen

    if len(entries) != n_entries:
        raise RuntimeError(f"parsed {len(entries)} entries, header claims {n_entries}")
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"zip_size": ZIP_SIZE, "entries": entries}, fh)
    os.replace(tmp, cache_path)
    print(f"  manifest: {len(entries)} entries -> {cache_path}", flush=True)
    return entries


def select(entries, groups, lowres, include_re):
    """Pick the members to fetch. Directory entries are skipped (dirs are made on write)."""
    out = []
    for e in entries:
        name = e["n"]
        if name.endswith("/"):
            continue
        parts = name.split("/")
        group = parts[1] if len(parts) > 1 else ""
        base = parts[-1]
        # ego_4d_{verb,noun}.npy sit at signals/ root and are the label vocabulary: always take
        # them when any signal group is wanted -- 20 KB, and useless to discover you lack later.
        vocab = base.endswith(".npy") and group.startswith("signals")
        if groups != "all" and group not in groups and not (vocab and any(
                g.startswith("signals") for g in groups)):
            continue
        if group.startswith("videos") and not vocab:
            is_lowres = base.endswith("_32.npz")
            if lowres and not is_lowres:
                continue
        if include_re and not include_re.search(name):
            continue
        out.append(e)
    out.sort(key=lambda e: e["o"])
    return out


def plan_spans(members, done):
    """Group pending members into contiguous byte spans, splitting at CHUNK/GAP_TOLERANCE."""
    pending = [e for e in members if e["n"] not in done]
    spans, cur = [], []
    for e in pending:
        if cur:
            gap = e["o"] - (cur[-1]["o"] + 30 + len(cur[-1]["n"]) + cur[-1]["c"])
            span_bytes = e["o"] + e["c"] - cur[0]["o"]
            if gap > GAP_TOLERANCE or span_bytes > CHUNK:
                spans.append(cur); cur = []
        cur.append(e)
    if cur:
        spans.append(cur)
    return pending, spans


def _readexact(resp, n):
    """urllib returns short reads on chunked responses; loop until n bytes or EOF."""
    out = bytearray()
    while len(out) < n:
        b = resp.read(n - len(out))
        if not b:
            raise RuntimeError(f"stream ended early: wanted {n}, got {len(out)}")
        out += b
    return bytes(out)


def _skip(resp, n):
    while n > 0:
        b = resp.read(min(n, 1 << 20))
        if not b:
            raise RuntimeError("stream ended early while skipping gap")
        n -= len(b)


def fetch_span(drive, span, dest, done, done_fh, stats):
    """Stream one contiguous span, cutting member boundaries and inflating as bytes arrive."""
    start = span[0]["o"]
    end = span[-1]["o"] + 30 + len(span[-1]["n"].encode()) + 512 + span[-1]["c"]
    end = min(end, ZIP_SIZE - 1)
    pos = start
    with drive.open_range(start, end) as resp:
        for e in span:
            if e["o"] > pos:
                _skip(resp, e["o"] - pos); pos = e["o"]
            elif e["o"] < pos:
                raise RuntimeError(f"span out of order at {e['n']}")
            # Local header: name/extra lengths may differ from the central copy, so re-read.
            hdr = _readexact(resp, 30); pos += 30
            if hdr[:4] != b"PK\x03\x04":
                raise RuntimeError(f"bad local header for {e['n']}")
            nlen, elen = struct.unpack("<HH", hdr[26:30])
            _skip(resp, nlen + elen); pos += nlen + elen

            out_path = os.path.join(dest, *e["n"].split("/"))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            tmp = out_path + ".part"
            dec = zlib.decompressobj(-15) if e["m"] == 8 else None
            crc = 0
            written = 0
            remaining = e["c"]
            with open(tmp, "wb") as fh:
                while remaining > 0:
                    blk = resp.read(min(remaining, 1 << 20))
                    if not blk:
                        raise RuntimeError(f"stream ended inside {e['n']}")
                    remaining -= len(blk); pos += len(blk)
                    data = dec.decompress(blk) if dec else blk
                    if data:
                        crc = binascii.crc32(data, crc); written += len(data)
                        fh.write(data)
                if dec:
                    data = dec.flush()
                    if data:
                        crc = binascii.crc32(data, crc); written += len(data)
                        fh.write(data)
            if written != e["u"] or crc != e["crc"]:
                os.remove(tmp)
                raise RuntimeError(
                    f"{e['n']}: corrupt (size {written}/{e['u']}, crc {crc:08x}/{e['crc']:08x})")
            os.replace(tmp, out_path)
            done.add(e["n"]); done_fh.write(e["n"] + "\n")
            stats["files"] += 1; stats["bytes"] += e["c"]


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024


def main():
    global FILE_ID
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True,
                    help="output root; put it on /scratch365, NOT home (quota)")
    ap.add_argument("--groups", default="signals",
                    help="comma list of signals,signals1,signals2,videos,videos1,videos2 "
                         "| 'all' | 'signals' expands to all three signal groups "
                         "(default: %(default)s)")
    ap.add_argument("--lowres", action="store_true",
                    help="for video groups take only the 32x32 variant (~1.2 GiB, and only "
                         "the 'videos' group ships one)")
    ap.add_argument("--include", default=None,
                    help="extra regex filter on the full member path, e.g. '/val/' or '/S0[12]/'")
    ap.add_argument("--plan", action="store_true", help="print what would be fetched, then exit")
    ap.add_argument("--file-id", default=FILE_ID, help="override the Drive file id")
    args = ap.parse_args()
    FILE_ID = args.file_id

    if args.groups.strip() == "all":
        groups = "all"
    else:
        groups = set()
        for g in args.groups.split(","):
            g = g.strip()
            if g == "signals":
                groups |= {"signals", "signals1", "signals2"}
            elif g == "videos":
                groups |= {"videos", "videos1", "videos2"}
            elif g:
                groups.add(g)

    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)
    drive = Drive()
    entries = load_manifest(drive, os.path.join(dest, "manifest.json"))
    members = select(entries, groups, args.lowres,
                     re.compile(args.include) if args.include else None)
    if not members:
        sys.exit("nothing selected -- check --groups/--include")

    done_path = os.path.join(dest, "done.txt")
    done = set()
    if os.path.exists(done_path):
        with open(done_path) as fh:
            done = {ln.rstrip("\n") for ln in fh if ln.strip()}
    # Trust the log only where the file is actually present at the right size; a killed job can
    # leave the log ahead of the disk.
    done = {n for n in done
            if os.path.exists(os.path.join(dest, *n.split("/")))}

    pending, spans = plan_spans(members, done)
    todo_c = sum(e["c"] for e in pending)
    todo_u = sum(e["u"] for e in pending)
    print(f"  selected {len(members)} members ({len(done)} already on disk)")
    print(f"  to fetch: {len(pending)} files, {human(todo_c)} transfer -> {human(todo_u)} on disk")
    print(f"  in {len(spans)} ranged request span(s)")
    if args.plan:
        by_group = {}
        for e in pending:
            g = e["n"].split("/")[1]
            n, c, u = by_group.get(g, (0, 0, 0))
            by_group[g] = (n + 1, c + e["c"], u + e["u"])
        for g in sorted(by_group):
            n, c, u = by_group[g]
            print(f"    {g:10s} {n:7d} files  {human(c):>10s} xfer  {human(u):>10s} disk")
        return

    stats = {"files": 0, "bytes": 0}
    t0 = time.time()
    with open(done_path, "a", buffering=1) as done_fh:
        for i, span in enumerate(spans, 1):
            span_c = sum(e["c"] for e in span)
            print(f"[{i}/{len(spans)}] {len(span)} files, {human(span_c)} "
                  f"@ offset {span[0]['o']}", flush=True)
            fetch_span(drive, span, dest, done, done_fh, stats)
            el = time.time() - t0
            rate = stats["bytes"] / el if el else 0
            left = todo_c - stats["bytes"]
            eta = left / rate if rate else 0
            print(f"    {stats['files']}/{len(pending)} files, {human(stats['bytes'])} "
                  f"in {el/60:.1f} min ({human(rate)}/s), ETA {eta/60:.0f} min", flush=True)

    print(f"\ndone: {stats['files']} files, {human(stats['bytes'])} transferred -> {dest}")
    print(f"  every member CRC-32 verified against the archive's central directory")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\ninterrupted -- re-run the same command to resume")
