"""Turn ParlGov's cabinet table into real parliaments for evaluation.

Source: ParlGov (Doering and Manow), development version, CC-BY-SA 4.0
    https://parlgov.org/data/parlgov-development_csv-utf-8/view_cabinet.csv
Download it with   bash scripts/parlgov_download.sh   (into data/parlgov/raw/).

Scope (spec OD4): post-1945 cabinets in 14 Western European parliamentary
democracies (13 remain: Spain had no coalition cabinets in this period)
excluding caretakers and single-party cabinets.  For each
cabinet we keep the parties with at least 1% of seats, at most the 8 largest
(the trained policies cover 3-8 parties); cabinets with a member outside that
set are dropped and counted.  ParlGov has no portfolio counts, so the real
outcome we compare with is the coalition itself and the prime minister's party.

Output: data/parlgov/cabinets.json (Western Europe, the spec's default scope);
``python scripts/parlgov_prepare.py all`` also writes cabinets_all.json with the
other parliamentary democracies in ParlGov.
"""
import csv
import json
import os
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "parlgov", "raw", "view_cabinet.csv")
OUT = os.path.join(ROOT, "data", "parlgov", "cabinets.json")
COUNTRIES = {"AUT", "BEL", "DNK", "FIN", "DEU", "ISL", "IRL", "ITA", "LUX", "NLD", "NOR",
             "PRT", "ESP", "SWE"}
MAX_PARTIES, MIN_SHARE = 8, 0.01
# Parliamentary systems in ParlGov outside Western Europe, for an out-of-sample
# extension (presidential systems and Switzerland's collegial executive excluded).
OTHER = {"AUS", "BGR", "CAN", "CYP", "CZE", "EST", "GRC", "HRV", "HUN", "ISR", "JPN", "LTU",
         "LVA", "MLT", "NZL", "POL", "ROU", "SVK", "SVN", "GBR", "FRA", "TUR"}


def main(scope="western"):
    countries = COUNTRIES if scope == "western" else COUNTRIES | OTHER
    out_path = OUT if scope == "western" else OUT.replace("cabinets.json", "cabinets_all.json")
    rows = defaultdict(list)
    with open(RAW, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["country_name_short"] in countries and r["start_date"] >= "1945-01-01":
                rows[r["cabinet_id"]].append(r)
    out, dropped = [], defaultdict(int)
    for cid, rs in rows.items():
        r0 = rs[0]
        if r0["caretaker"] == "1":
            dropped["caretaker"] += 1
            continue
        try:
            house = int(float(r0["election_seats_total"]))
        except ValueError:
            dropped["no house size"] += 1
            continue
        parties = []
        for r in rs:
            try:
                seats = int(float(r["seats"] or 0))
            except ValueError:
                seats = 0
            parties.append(dict(name=r["party_name_short"], seats=seats,
                                cab=r["cabinet_party"] == "1", pm=r["prime_minister"] == "1"))
        members = [p for p in parties if p["cab"]]
        if len(members) < 2:
            dropped["single-party cabinet"] += 1
            continue
        kept = sorted([p for p in parties if p["seats"] >= MIN_SHARE * house],
                      key=lambda p: -p["seats"])[:MAX_PARTIES]
        if any(m not in kept for m in members) or len(kept) < 3:
            dropped["member outside the 8 largest / tiny party"] += 1
            continue
        pm = [i for i, p in enumerate(kept) if p["pm"]]
        if len(pm) != 1:
            dropped["prime minister not identified"] += 1
            continue
        q = house // 2 + 1
        if max(p["seats"] for p in kept) >= q:
            dropped["single-party majority in parliament"] += 1
            continue
        cab_seats = sum(p["seats"] for p in kept if p["cab"])
        out.append(dict(
            cabinet_id=int(cid), country=r0["country_name_short"], start=r0["start_date"],
            name=r0["cabinet_name"], house=house, quota=q,
            parties=[p["name"] for p in kept], seats=[p["seats"] for p in kept],
            coalition=[int(p["cab"]) for p in kept], pm=pm[0],
            majority=cab_seats >= q))
    out.sort(key=lambda c: (c["country"], c["start"]))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=0)
    maj = sum(c["majority"] for c in out)
    print(f"{len(out)} coalition cabinets kept ({maj} majority, {len(out) - maj} minority) "
          f"from {len(set(c['country'] for c in out))} countries")
    for k, v in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"  dropped {v:4d}: {k}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "western")
