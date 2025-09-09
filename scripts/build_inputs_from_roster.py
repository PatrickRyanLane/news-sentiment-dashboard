#!/usr/bin/env python3
import csv, os, sys

ROSTER = "data/roster.csv"

OUT_CEOS_TXT     = "ceos.txt"            # used by your CEO pipeline
OUT_CEO_ALIASES  = "ceo_aliases.csv"     # brand=CEO name, alias=Company (for narrowing)

def main():
    if not os.path.exists(ROSTER):
        print(f"ERROR: {ROSTER} not found", file=sys.stderr)
        sys.exit(1)

    ceos = []
    pairs = []  # (brand, alias)

    with open(ROSTER, newline='', encoding="utf-8") as f:
        rd = csv.DictReader(f)
        need = {"CEO","Company","Website","Stock","Sector"}
        missing = need - set(rd.fieldnames or [])
        if missing:
            print(f"ERROR: roster missing columns: {', '.join(sorted(missing))}", file=sys.stderr)
            sys.exit(2)

        for row in rd:
            ceo = (row["CEO"] or "").strip()
            company = (row["Company"] or "").strip()
            if not ceo or not company:
                continue
            ceos.append(ceo)
            # keep query narrow: ("CEO" OR "Company")
            pairs.append((ceo, company))

    # ceos.txt (one per line)
    with open(OUT_CEOS_TXT, "w", encoding="utf-8") as f:
        for c in sorted(set(ceos)):
            f.write(c + "\n")

    # ceo_aliases.csv  brand,alias
    with open(OUT_CEO_ALIASES, "w", newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["brand", "alias"])
        for b,a in sorted(set(pairs)):
            w.writerow([b, a])

    print(f"Wrote {OUT_CEOS_TXT} and {OUT_CEO_ALIASES} from {ROSTER}")

if __name__ == "__main__":
    main()
