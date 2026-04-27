from pathlib import Path
import pandas as pd


def build_mask_input(
    basins_dir: Path,
    out_fp: Path,
    overwrite: bool = False,
) -> Path:
    basins_dir = Path(basins_dir)
    out_fp = Path(out_fp)

    if out_fp.exists() and not overwrite:
        print(f"[SKIP] mask input exists: {out_fp}")
        return out_fp

    rows = []

    for fp in sorted(basins_dir.rglob("*.json")):
        site_id = fp.stem

        # Expected structure can be:
        # basins_json/STATE/XX/XXXX/site.json
        # or basins_json/STATE/site.json
        try:
            state = fp.relative_to(basins_dir).parts[0].upper()
        except Exception:
            continue

        rows.append(
            {
                "site_id": site_id,
                "state": state,
                "path": str(fp.resolve()),
            }
        )

    if not rows:
        raise RuntimeError(f"No basin JSON files found under {basins_dir}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["state", "site_id"]).reset_index(drop=True)

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, sep="\t", index=False)

    print(f"[DONE] saved mask input: {out_fp}")
    print(f"basins: {len(df)}")
    print(f"states: {df['state'].nunique()}")

    return out_fp