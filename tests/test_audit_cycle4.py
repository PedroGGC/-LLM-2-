import json
import zipfile
import pytest
from llm2.research import audit_cycle4


def test_archive_metadata_cannot_inject_git_options(tmp_path,monkeypatch):
    revision="--output=outside.zip"
    campaign={"status":"complete","engineering_smoke":False,"tests_skipped":False,
        "protocol_id":audit_cycle4.PROTOCOL_ID,"test_evaluated":False,
        "provenance":{"git":{"commit":revision}}}
    results=tmp_path/"results.zip";weights=tmp_path/"checkpoints.zip"
    with zipfile.ZipFile(results,"w") as z:
        z.writestr("campaign.json",json.dumps(campaign));z.writestr("report/report.json","{}")
        z.writestr("bootstrap.json",json.dumps({"source_git_revision":revision}))
    with zipfile.ZipFile(weights,"w") as z:z.writestr("campaign.json",json.dumps(campaign))
    monkeypatch.setattr(audit_cycle4.subprocess,"check_output",lambda *a,**k:pytest.fail("Git must not run"))
    with pytest.raises(ValueError,match="exact Git revision"):
        audit_cycle4.audit(results,weights,tmp_path/"audit")
