"""Independent archive audit of Cycle 4; reads checkpoint hashes, never pickle."""
import argparse
import hashlib
import io
import json
import math
import re
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import zipfile

import yaml
from .cycle4_protocol import ROOT, PROTOCOL_ID, SEEDS, VARIANTS, configuration
from .cycle2_protocol import configuration_stats
from .cycle3_protocol import DATA_HASHES
from .cycle4_report import analyze


def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""):digest.update(block)
    return digest.hexdigest()


def member_sha(archive,name):
    digest=hashlib.sha256()
    with archive.open(name) as stream:
        for block in iter(lambda:stream.read(1024**2),b""):digest.update(block)
    return digest.hexdigest()


def validate_archive(archive):
    names=archive.namelist()
    assert len(names)==len(set(names)),"duplicate ZIP entries"
    assert all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts
               and "\\" not in n for n in names),"unsafe member"
    assert archive.testzip() is None,"CRC mismatch"


def json_member(archive,name):return json.loads(archive.read(name))


def windows(archive,name,expected_tokens):
    sums=[];tokens=0;count=0
    with archive.open(name) as stream:
        for index,line in enumerate(stream):
            row=json.loads(line)
            assert row["window_index"]==index and row["target_start_offset"]==index*256+1
            assert row["input_length"]==row["supervised_tokens"]==256
            assert math.isfinite(row["nll_sum"]) and row["nll_sum"]>=0
            sums.append(row["nll_sum"]);tokens+=row["supervised_tokens"];count+=1
    assert tokens==expected_tokens
    return math.fsum(sums)/tokens,count


def audit(results,checkpoints,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    counts={"git_files":0,"snapshots":0,"snapshot_members":0,"checkpoint_hashes":0,"full_evaluations":0,"depth_evaluations":0}
    errors=[]
    with zipfile.ZipFile(results) as z,zipfile.ZipFile(checkpoints) as weights:
        validate_archive(z);validate_archive(weights)
        campaign=json_member(z,"campaign.json");report=json_member(z,"report/report.json")
        bootstrap=json_member(z,"bootstrap.json")
        assert campaign["status"]=="complete" and not campaign["engineering_smoke"] and not campaign["tests_skipped"]
        assert campaign["protocol_id"]==PROTOCOL_ID and not campaign["test_evaluated"]
        assert json_member(weights,"campaign.json")==campaign
        revision=bootstrap["source_git_revision"]
        if not isinstance(revision,str) or not re.fullmatch(r"[0-9a-f]{40}",revision):
            raise ValueError("archive must declare an exact Git revision, never command options")
        assert campaign["provenance"]["git"]["commit"]==revision
        archive=subprocess.check_output(["git","archive","--format=zip",revision],cwd=ROOT)
        with zipfile.ZipFile(io.BytesIO(archive)) as git:
            for name,digest in bootstrap["source_files_sha256"].items():
                assert hashlib.sha256(git.read(name)).hexdigest()==digest,name
                counts["git_files"]+=1
            expected_source=campaign["provenance"]["source_sha256"]
            expected_configs=campaign["provenance"]["versioned_configs_sha256"]
            for name,digest in expected_source.items():assert hashlib.sha256(git.read(name)).hexdigest()==digest
            for name in z.namelist():
                if name.endswith("/source.zip"):
                    with zipfile.ZipFile(io.BytesIO(z.read(name))) as snapshot:
                        expected={**expected_source,**{"configs/"+k:v for k,v in expected_configs.items()}}
                        assert set(snapshot.namelist())==set(expected),name
                        for member,digest in expected.items():assert member_sha(snapshot,member)==digest,(name,member)
                        counts["snapshots"]+=1;counts["snapshot_members"]+=len(expected)
        dataset=json_member(z,"data/manifest.json")
        assert dataset["files"]==DATA_HASHES
        assert hashlib.sha256(z.read("data/manifest.json")).hexdigest()==campaign["dataset_sha256"]
        assert member_sha(z,"data/tokenizer.json")==DATA_HASHES["tokenizer.json"]
        assert weights.read("data/manifest.json")==z.read("data/manifest.json")
        expected_tokens=(dataset["splits"]["val"]["tokens"]-1)//256*256
        assert expected_tokens==1144064
        assert {(j["variant"],j["seed"]) for j in campaign["runs"]}=={(v,s) for v in VARIANTS for s in SEEDS}
        assert len(campaign["runs"])==18
        curves=[];rows=[];initial={};environment=None
        for job in campaign["runs"]:
            identity=job["id"];base="runs/"+identity
            cfg=configuration(job["variant"],campaign["dataset_path"],job["directory"],job["seed"])
            assert cfg.to_dict()==job["config_resolved"]==yaml.safe_load(z.read("configs/"+identity+".yaml"))
            stats=configuration_stats(cfg)
            assert stats["registered_parameters"]==stats["parameters_formula"]==job["registered_parameters"]
            assert stats["step_flops_estimated"]==job["step_flops_estimated"]
            run=json_member(z,base+"/run.json");summary=json_member(z,base+"/summary.json")
            environment=environment or run["environment"]
            assert run["environment"]["cuda_visible_devices"]==job["physical_gpu"]
            assert run["environment"]["device"]=="Tesla T4"
            assert run["config"]==cfg.to_dict() and run["provenance"]["source_sha256"]==expected_source
            assert summary["step"]==6000 and summary["stop_reason"]=="max_steps"
            assert summary["tokens_processed"]==summary["input_tokens_processed"]==49152000
            assert summary["training_flops_estimate"]==6000*stats["step_flops_estimated"]
            assert summary["step_auxiliary_flops_estimate"]==summary["curriculum_probe_flops_estimate"]==0
            if cfg.research.paired_common_init:
                p=run["paired_init_provenance"]
                assert p["tied_embedding_lm_head"]
                value=p["common_combined_sha256"]
                assert initial.setdefault(job["seed"],value)==value==job["initial_common_sha256"]
            for point in job["checkpoint_evaluations"]:
                step=point["step"];folder=base+f"/evaluations/step-{step:08d}"
                language=json_member(z,folder+"/language.json")
                assert language["evaluation_split"]=="val" and language["validation_scope"]=="all_full_windows"
                assert language["dataset_sha256"]==campaign["dataset_sha256"] and language["validation_tokens"]==expected_tokens
                value,n=windows(z,folder+"/windows.jsonl",expected_tokens)
                error=abs(value-language["validation_loss"]);errors.append(error)
                assert error<1e-6 and abs(value-point["validation_loss"])<1e-6
                meta=json_member(z,base+f"/checkpoints/step-{step:08d}/metadata.json")
                assert meta["config"]==cfg.to_dict() and meta["dataset_sha256"]==campaign["dataset_sha256"]
                assert meta["provenance"]["source_sha256"]==expected_source
                assert meta["files"]["model.safetensors"]==language["checkpoint_model_sha256"]
                curves.append({"variant":job["variant"],"seed":job["seed"],"step":step,
                    "validation_loss":language["validation_loss"],"validation_tokens":expected_tokens,
                    "training_targets":step*8192,"training_flops_estimated":step*job["step_flops_estimated"]})
                counts["full_evaluations"]+=1
            assert [p["step"] for p in job["checkpoint_evaluations"]]==[1500,3000,4500,6000]
            final=base+"/checkpoints/step-00006000"
            assert weights.read(final+"/metadata.json")==z.read(final+"/metadata.json")
            for name,digest in json_member(weights,final+"/metadata.json")["files"].items():
                assert member_sha(weights,final+"/"+name)==digest,(identity,name)
                counts["checkpoint_hashes"]+=1
            row=next(r for r in report["rows"] if (r["variant"],r["seed"])==(job["variant"],job["seed"]))
            assert row["validation_loss"]==job["checkpoint_evaluations"][-1]["validation_loss"]
            assert row["checkpoint_model_sha256"]==job["checkpoint_model_sha256"]
            assert row["training_targets"]==summary["tokens_processed"]
            rows.append(row)
        for name in z.namelist():
            if name.endswith("/depth/depth.json"):
                depth=json_member(z,name);identity=name.split("/")[1]
                job=next(j for j in campaign["runs"] if j["id"]==identity)
                assert depth["checkpoint_model_sha256"]==job["checkpoint_model_sha256"]
                assert depth["dataset_sha256"]==campaign["dataset_sha256"] and not depth["test_evaluated"]
                assert [r["iterations_override"] for r in depth["rows"]]==[1,2,4,8]
                for row in depth["rows"]:
                    value,n=windows(z,name.removesuffix("depth.json")+f"k-{row['iterations_override']}-windows.jsonl",expected_tokens)
                    errors.append(abs(value-row["validation_loss"]));assert errors[-1]<1e-6
                    counts["depth_evaluations"]+=1
        recalculated=analyze(rows,curves,report["isolated_profiles"])
        assert curves==report["curves"]
        for name in ("contrasts","quality_component_candidate","simplifications","complete_seeds"):
            assert recalculated[name]==report[name],name
        assert len(report["isolated_profiles"])==12
        profiles=[]
        for p in report["isolated_profiles"]:
            original=json_member(z,f"isolated/{p['variant']}-gpu-{p['gpu_slot']}/profile.json")
            durations=[s["step_train_wall_seconds"] for s in original["unprofiled_samples"]]
            assert len(durations)==90
            median=statistics.median(statistics.median(durations[i:i+30]) for i in range(0,90,30))
            assert median==p["median_step_seconds"]
            assert p["energy"]==original["energy"]
            profiles.append(p)
        summaries={v:{"mean_nll":statistics.mean(r["validation_loss"] for r in rows if r["variant"]==v),
            "isolated_ms_by_gpu":{p["gpu_slot"]:p["median_step_seconds"]*1000 for p in profiles if p["variant"]==v},
            "energy_microjoules_per_target_by_gpu":{p["gpu_slot"]:p["energy"]["gpu_joules_per_training_target"]*1e6 for p in profiles if p["variant"]==v}}
            for v in VARIANTS}
    result={"status":"verified","results":{"sha256":sha_file(results),"bytes":Path(results).stat().st_size},
        "checkpoints":{"sha256":sha_file(checkpoints),"bytes":Path(checkpoints).stat().st_size},
        "revision":revision,"dataset_sha256":campaign["dataset_sha256"],"counts":counts,
        "max_reconstructed_nll_error":max(errors),"environment":environment,"summary":summaries,
        "contrasts":recalculated["contrasts"],"simplifications":recalculated["simplifications"],
        "session_budget":campaign["session_budget"],"limitations":"Weights/state hashes only; no pickle load or independent GPU rerun; exploratory three seeds, no reserved test."}
    (output/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"counts":counts,"max_nll_error":max(errors),
                      "checkpoints_sha256":result["checkpoints"]["sha256"]},indent=2))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--results",required=True);p.add_argument("--checkpoints",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();audit(a.results,a.checkpoints,a.output)
