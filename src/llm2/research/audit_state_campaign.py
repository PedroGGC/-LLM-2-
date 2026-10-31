"""Read a result bundle without extraction/pickle; separate speed from qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import zipfile


def audit(path):
    path=Path(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path) as z:
        if (len(z.infolist())>1000 or len(z.namelist())!=len(set(z.namelist()))
                or sum(i.file_size for i in z.infolist())>128*1024**2 or z.testzip() is not None):
            raise ValueError("invalid/oversized result bundle")
        c=json.loads(z.read("campaign.json"));p=json.loads(z.read("platforms.json"))
        rows=[]
        for value in p["rows"]:
            rule,slot=value["rule"],value["slot"]
            log=z.read(f"logs/qualify-{rule}-{slot}.log").decode("utf-8","replace")
            timeout="candidate exceeded 600 seconds" in log
            first=value.get("first_step_seconds")
            rows.append({k:value.get(k) for k in ("rule","slot","status","median_step_seconds",
                "first_step_seconds","warmup_seconds","peak_vram_bytes","parameters","state_bytes_per_sequence")})
            rows[-1].update(timeout_in_log=timeout,first_step_exceeded_600s=first is not None and first>600,
                correctness_reported=value.get("correctness",{}).get("passed"),
                supervisor_verified=value.get("timeout_enforced_by")=="parent_process")
        grouped={rule:{"mean_median_step_ms":1000*statistics.mean(r["median_step_seconds"] for r in rows if r["rule"]==rule),
                       "timeout_affected_slots":[r["slot"] for r in rows if r["rule"]==rule and r["timeout_in_log"]]}
                 for rule in {r["rule"] for r in rows}}
    affected=any(r["timeout_in_log"] or r["first_step_exceeded_600s"] for r in rows)
    return {"zip_sha256":digest,"zip_bytes":path.stat().st_size,"status_reported":c["status"],
        "source_revision":c["provenance"]["git"]["commit"],"protocol_id":c["protocol_id"],
        "environment":p["environment"],"reported_selected_rule":p["selected_rule"],
        "scientific_training_jobs":len(c["runs"]),"session_budget":c.get("session_budget"),
        "rows":rows,"grouped":grouped,"disk_footprint_bytes":None,
        "qualification_decision":"requalify_external_timeout" if affected else "inspect_source_and_supervisor_contract",
        "limitations":["not a language quality experiment","common-shell memory rules, not full published architectures",
                        "disk footprint outside ZIP was not recorded","timeout can be swallowed/retried by compiler; log does not establish final fallback backend"]}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("archive");p.add_argument("--output",required=True)
    a=p.parse_args(argv)
    result=audit(a.archive)
    with Path(a.output).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,indent=2,ensure_ascii=False);f.write("\n")
    print(result["qualification_decision"])


if __name__=="__main__":main()
