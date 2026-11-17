#!/usr/bin/env python
"""Create / fill the Zenodo record of the HAZBRA dataset (never publishes: you press "Publish" on zenodo.org).

Token: a Zenodo personal access token with scopes deposit:write and deposit:actions, read from
$ZENODO_TOKEN or from ~/.zenodo_token (chmod 600).  State (record id, bucket, DOI) is kept in
<staging>/../zenodo_state.json so the steps can be rerun.

    python zenodo_upload.py create            # draft record + metadata + reserved DOI (nothing public)
    python zenodo_upload.py upload [--dir D]  # upload every file of the staging folder (skips files already there)
    python zenodo_upload.py metadata          # re-send metadata.json to the draft
    python zenodo_upload.py status            # files in the draft, DOI, link
    python zenodo_upload.py newversion        # after publishing: open a draft of the next version (same concept DOI)
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import requests

API = "https://zenodo.org/api"
HERE = Path(__file__).resolve().parent
STAGING = Path("/gws/ssde/j25a/craft/lgcatt/z_GIZ/HAZBRA_zenodo/staging")
STATE = STAGING.parent / "zenodo_state.json"


def token() -> str:
    t = os.environ.get("ZENODO_TOKEN") or (Path.home() / ".zenodo_token").read_text().strip()
    if not t:
        sys.exit("no Zenodo token ($ZENODO_TOKEN or ~/.zenodo_token)")
    return t


def s() -> requests.Session:
    ses = requests.Session()
    ses.headers["Authorization"] = f"Bearer {token()}"
    return ses


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2))


def metadata() -> dict:
    return json.loads((HERE / "metadata.json").read_text())


def check(r: requests.Response) -> dict:
    if r.status_code >= 400:
        sys.exit(f"Zenodo error {r.status_code}: {r.text[:500]}")
    return r.json() if r.content else {}


def create(_):
    st = load_state()
    if st.get("id"):
        sys.exit(f"draft already exists: {st['html']} (use 'status')")
    md = metadata()
    md["prereserve_doi"] = True
    d = check(s().post(f"{API}/deposit/depositions", json={"metadata": md}))
    st = dict(id=d["id"], bucket=d["links"]["bucket"], html=d["links"]["html"],
              doi=d["metadata"]["prereserve_doi"]["doi"])
    save_state(st)
    print(f"draft created: {st['html']}\nreserved DOI: {st['doi']} (becomes active only when you publish)")


def send_metadata(_):
    st = load_state()
    md = metadata()
    md["prereserve_doi"] = True
    check(s().put(f"{API}/deposit/depositions/{st['id']}", json={"metadata": md}))
    print("metadata updated")


def remote_files(ses, st) -> dict:
    d = check(ses.get(f"{API}/deposit/depositions/{st['id']}"))
    return {f["filename"]: f for f in d.get("files", [])}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def upload(a):
    st = load_state()
    ses = s()
    have = remote_files(ses, st)
    files = sorted(p for p in Path(a.dir).iterdir() if p.is_file() and ".tmp" not in p.name)
    for p in files:
        local = md5(p)
        if p.name in have and have[p.name]["checksum"] == local:
            print(f"[skip] {p.name} (already uploaded)")
            continue
        if p.name in have:                                    # changed file: replace
            check(ses.delete(have[p.name]["links"]["self"]))
        print(f"[upload] {p.name} ({p.stat().st_size / 1e9:.2f} GB)", flush=True)
        with open(p, "rb") as f:
            r = check(ses.put(f"{st['bucket']}/{p.name}", data=f))
        if r.get("checksum", "").replace("md5:", "") != local:
            sys.exit(f"checksum mismatch for {p.name}")
    print(f"done: {len(files)} files in {st['html']}")


def status(_):
    st = load_state()
    if not st:
        sys.exit("no draft yet (run 'create')")
    have = remote_files(s(), st)
    tot = sum(f["filesize"] for f in have.values())
    print(f"{st['html']}\nDOI {st['doi']}\n{len(have)} files, {tot / 1e9:.2f} GB")
    for n, f in sorted(have.items()):
        print(f"  {n:45s} {f['filesize'] / 1e9:7.2f} GB")


def newversion(_):
    st = load_state()
    d = check(s().post(f"{API}/deposit/depositions/{st['id']}/actions/newversion"))
    new = check(s().get(d["links"]["latest_draft"]))
    save_state(dict(id=new["id"], bucket=new["links"]["bucket"], html=new["links"]["html"],
                    doi=new["metadata"].get("prereserve_doi", {}).get("doi", ""), previous=st))
    print(f"new version draft: {new['links']['html']} (files of the previous version are copied; "
          f"'upload' replaces the changed ones)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("create").set_defaults(fn=create)
    sub.add_parser("metadata").set_defaults(fn=send_metadata)
    u = sub.add_parser("upload")
    u.add_argument("--dir", default=str(STAGING))
    u.set_defaults(fn=upload)
    sub.add_parser("status").set_defaults(fn=status)
    sub.add_parser("newversion").set_defaults(fn=newversion)
    a = ap.parse_args()
    a.fn(a)
