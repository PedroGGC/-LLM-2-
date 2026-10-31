"""Combine explicitly supplied state-study reports; never discover arbitrary ZIP history."""
import argparse
import json
from pathlib import Path
import zipfile
from ..data import write_json
from .state_protocol import PROTOCOL_ID
from .state_report import analyze


def merge(paths,output):
    reports=[]
    for name in paths:
        path=Path(name)
        if path.suffix.lower()==".zip":
            with zipfile.ZipFile(path) as z:
                if z.namelist().count("report/report.json")!=1:raise ValueError("missing/duplicate report")
                if z.getinfo("report/report.json").file_size>16*1024**2:raise ValueError("oversized report")
                reports.append(json.loads(z.read("report/report.json")))
        else:reports.append(json.loads(path.read_text(encoding="utf-8")))
    if not reports:raise ValueError("explicit reports required")
    def identity(r):
        i=r["identity"];env=i["environment"]
        return {"rule":i["selected_rule"],"source":i["source_sha256"],"data":i["dataset_sha256"],
            "lr":i["learning_rates"],"python":env["python"],"packages":env["packages"],
            "device":env["device"],"cuda":env["cuda_version"],"smoke":r["engineering_smoke"]}
    if any(r.get("protocol_id")!=PROTOCOL_ID or identity(r)!=identity(reports[0]) for r in reports):
        raise ValueError("incompatible study reports")
    rows=[row for r in reports for row in r["rows"]]
    # Duplicates are fatal, including identical replicas. Never increase effective n.
    benchmarks=next((r.get("isolated_profiles",[]) for r in reports if len(r.get("isolated_profiles",[]))==8),[])
    result=analyze(rows,benchmarks,smoke=reports[0]["engineering_smoke"])
    result["curves"]=[point for r in reports for point in r.get("curves",[])]
    result["benchmark_scope"]="first complete isolated benchmark series; not pooled across sessions"
    result["identity"]=reports[0]["identity"];result["input_reports"]=[str(p) for p in paths]
    result["limitations"]+=" Isolated benchmark repetitions across sessions are not pooled as independent samples."
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    write_json(output,result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("reports",nargs="+");p.add_argument("--output",required=True)
    a=p.parse_args(argv);return merge(a.reports,a.output)


if __name__=="__main__":main()
