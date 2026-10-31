import json
from llm2.research.state_study import main
from test_training import training_config


def test_real_two_lane_state_study_smoke(training_config,tmp_path):
    output=tmp_path/"state-study"
    result=main(["--execute","--phase","experiment","--device","cpu","--engineering-smoke",
        "--data",training_config.data.path,"--output",str(output),"--seeds","241"])
    assert result["status"]=="complete"
    assert len(result["runs"])==4 and all(j["status"]=="complete" for j in result["runs"])
    assert len(result["tuning"])==8 and len(result["isolated_profiles"])==8
    report=json.loads((output/"report"/"report.json").read_text())
    assert report["engineering_smoke"] and not report["test_evaluated"]
    assert (tmp_path/"state-study-results.zip").exists()
    assert (tmp_path/"state-study-checkpoints.zip").exists()
    assert (output/"report"/"curves.csv").exists()
    assert (output/"report"/"quality_by_cost.png").exists()
    assert not list(tmp_path.glob("*-midway-*.zip"))
