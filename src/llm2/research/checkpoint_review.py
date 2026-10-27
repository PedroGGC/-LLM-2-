"""Evaluate saved checkpoints with their original source; never trains or unpickles state."""

import csv
import json
import math
from pathlib import Path
import shutil
import traceback
import zipfile

import torch

from ..checkpoint import read_checkpoint_metadata
from ..data import PreparedData, sha256_file
from ..runtime import environment, save_source_snapshot
from .campaign import Campaign, export_campaign, worker_environment
from .presentation import MODEL_NAMES, legend, number, review_text, stage


def verified_source(root, expected):
    root = Path(root).resolve()
    paths = list((root / "src").rglob("*.py")) + [root / "pyproject.toml"]
    actual = {p.relative_to(root).as_posix(): sha256_file(p) for p in paths if p.is_file()}
    if actual != expected:
        raise ValueError("O código de avaliação difere do código original dos checkpoints. "
                         "Restaure o checkout original; as verificações não serão relaxadas.")
    return actual


def verified_result(destination, metadata, data):
    result = json.loads((destination / "language.json").read_text(encoding="utf-8"))
    seq_len = metadata["config"]["data"]["seq_len"]
    count = (len(data.arrays["val"])-1)//seq_len
    if (result.get("validation_scope") != "all_full_windows" or result.get("dataset_sha256") != data.fingerprint
            or result.get("checkpoint_model_sha256") != metadata["files"]["model.safetensors"]
            or result.get("seq_len") != seq_len or result.get("precision") != "fp32"
            or result.get("validation_tokens") != count*seq_len):
        raise ValueError("Avaliação tem pesos, dados, precisão ou escopo incompatíveis.")
    windows = [json.loads(line) for line in (destination / "windows.jsonl").read_text(encoding="utf-8").splitlines()]
    if (len(windows) != count or any(
            row.get("window_index") != i or row.get("supervised_tokens") != seq_len
            or row.get("target_start_offset") != i*seq_len+1 for i, row in enumerate(windows))):
        raise ValueError("Janelas da avaliação estão incompletas ou fora de ordem.")
    loss = result.get("validation_loss")
    reconstructed = math.fsum(row["nll_sum"] for row in windows)/(count*seq_len)
    if not (loss is not None and math.isfinite(loss) and math.isfinite(reconstructed)
            and abs(loss-reconstructed) < 1e-6):
        raise ValueError("Loss agregada não corresponde às janelas de validação.")
    return result


def review_checkpoints(args):
    if not args.execute:
        return {"status": "not_started", "next": "Use --execute para avaliar; nenhum treino será iniciado."}
    if not args.review_steps or len(set(args.review_steps)) != len(args.review_steps) or min(args.review_steps) <= 0:
        raise ValueError("Escolha passos salvos positivos, sem duplicação.")
    original = Path(args.review_campaign).resolve()
    if not (original / "campaign.json").is_file():
        raise FileNotFoundError("Não encontrei a pasta da campanha anterior. Informe a pasta restaurada dos outputs; "
                                "o ZIP de resultados sozinho não contém os pesos.")
    manifest = json.loads((original / "campaign.json").read_text(encoding="utf-8"))
    root = Path(args.evaluation_root or manifest["root"]).resolve()
    legend()
    stage(1, 4, "LOCALIZAR A CAMPANHA ANTERIOR — SOMENTE AVALIAÇÃO, SEM TREINO")
    data_root = Path(args.data) if args.data else original / "data"
    if not (data_root / "manifest.json").is_file():
        raise FileNotFoundError("Não encontrei os dados preparados originais. Restaure a pasta data dos outputs "
                                "ou informe o caminho em LLM2_SETTINGS['data'].")
    data = PreparedData(data_root)
    if data.fingerprint != manifest["dataset_sha256"]:
        raise ValueError("Os dados não correspondem ao fingerprint da campanha original.")
    if args.device == "cuda" and (not torch.cuda.is_available() or not args.gpus
                                  or any(not g.isdecimal() or int(g) >= torch.cuda.device_count() for g in args.gpus)):
        raise ValueError("Ative a GPU no Kaggle e selecione um índice disponível.")
    expected = manifest["provenance"]["source_sha256"]
    verified_source(root, expected)
    tasks = []
    for job in manifest["runs"]:
        run_id = job["id"]
        if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
            raise ValueError("Identificador de execução inválido no manifest.")
        for step in sorted(args.review_steps):
            checkpoint = original / "runs" / run_id / "checkpoints" / f"step-{step:08d}"
            meta_path = checkpoint / "metadata.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint}. Restaure os outputs salvos do Kaggle.")
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            if any(not (checkpoint / name).is_file() for name in raw["files"]):
                raise FileNotFoundError(f"Faltam pesos/estado em {checkpoint}. O ZIP de análise contém só metadados; "
                                        "use a versão do notebook com os outputs completos.")
            metadata = read_checkpoint_metadata(checkpoint)
            if (metadata["dataset_sha256"] != data.fingerprint
                    or metadata["provenance"]["source_sha256"] != expected
                    or metadata["config"]["train"]["seed"] != job["seed"]):
                raise ValueError("Código, dados ou seed do checkpoint diferem da campanha original.")
            tasks.append((job, step, checkpoint, metadata))
    if not tasks:
        raise ValueError("A campanha anterior não contém execuções para avaliar.")
    print(f"✓ Dados, código e {len(tasks)} checkpoints conferidos. Nenhum treino será executado.", flush=True)
    from .naming import numbered_output
    output = Path(args.output).resolve() if args.output else numbered_output(original.parent, "llm2-review")
    campaign = Campaign(root, output, device=args.device, gpus=args.gpus)
    campaign.manifest.update(kind="checkpoint_review", original_campaign=str(original),
        dataset_sha256=data.fingerprint, evaluation_root=str(root), evaluation_source_sha256=expected,
        evaluation_environment=environment(), status="evaluating")
    failure, rows = None, []
    try:
        (output / "runner").mkdir()
        save_source_snapshot(output / "runner/source.zip", campaign.manifest["provenance"])
        (output / "evaluation_code").mkdir()
        with zipfile.ZipFile(output / "evaluation_code/source.zip", "x", compression=zipfile.ZIP_DEFLATED) as source:
            for name in expected:
                source.write(root / name, arcname=name)
        shutil.copyfile(data.root / "manifest.json", output / "dataset.manifest.json")
        shutil.copyfile(data.root / "tokenizer.json", output / "tokenizer.json")
        stage(2, 4, "COMPARAR CHECKPOINTS — LOSS MENOR = MELHOR")
        gpu = campaign.gpus[0]
        print(f"Avaliação sequencial em {'GPU física '+str(worker_environment(root, gpu).get('CUDA_VISIBLE_DEVICES')) if gpu is not None else 'CPU'}.", flush=True)
        for index, (job, step, checkpoint, metadata) in enumerate(tasks, 1):
            print(f"\n[{index}/{len(tasks)}] {MODEL_NAMES.get(job['model'], job['model'])} / "
                  f"seed {job['seed']} / passo {step}", flush=True)
            destination = output / "evaluations" / job["id"] / checkpoint.name
            destination.mkdir(parents=True)
            shutil.copyfile(checkpoint / "metadata.json", destination / "checkpoint.metadata.json")
            old = original / "runs" / job["id"] / "evaluations" / checkpoint.name
            reused = (old / "language.json").is_file() and (old / "windows.jsonl").is_file()
            if reused:
                verified_result(old, metadata, data)
                for name in ("language.json", "windows.jsonl"):
                    shutil.copyfile(old / name, destination / name)
                print("  Avaliação completa existente conferida e reutilizada.", flush=True)
            else:
                campaign.command(["-m", "llm2", "evaluate", "--checkpoint", checkpoint,
                    "--data", data.root, "--device", args.device, "--precision", "fp32",
                    "--output", destination / "language.json", "--windows-output", destination / "windows.jsonl"],
                    f"{job['id']}-{checkpoint.name}-evaluate.log", gpu=gpu)
            result = verified_result(destination, metadata, data)
            row = {"model": job["model"], "seed": job["seed"], "step": step,
                   "checkpoint": str(checkpoint), "checkpoint_model_sha256": metadata["files"]["model.safetensors"],
                   "validation_loss": result["validation_loss"], "validation_tokens": result["validation_tokens"],
                   "reused": reused, "evaluation": str(destination / "language.json")}
            rows.append(row)
            campaign.manifest["runs"].append({**row, "status": "complete"})
            campaign.save()
            print(f"  ✓ Loss completa: {number(result['validation_loss'], 6)} (menor = melhor); "
                  f"{number(result['validation_tokens'], 0)} alvos avaliados.", flush=True)
        if len({r["validation_tokens"] for r in rows}) != 1:
            raise ValueError("Os checkpoints não foram avaliados na mesma quantidade de alvos.")
        stage(3, 4, "RESULTADO — QUAL CHECKPOINT PREVIU MELHOR OS TOKENS?")
        text = review_text(rows)
        (output / "resumo-pt.md").write_text(text, encoding="utf-8")
        with (output / "checkpoints.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(text, flush=True)
        campaign.manifest["status"] = "complete"
    except Exception as exc:
        failure = exc
        campaign.abort()
        campaign.manifest.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    finally:
        stage(4, 4, "EXPORTAR RESULTADOS" if failure is None else "EXPORTAR RESULTADOS PARCIAIS — HOUVE UMA FALHA")
        campaign.save()
        archive = export_campaign(output)
        print(f"ZIP para enviar à análise: {archive}", flush=True)
    if failure is not None:
        raise RuntimeError(f"Avaliação interrompida. Logs e resultados parciais: {archive}") from failure
    return {"status": "complete", "results_zip": str(archive), "campaign": str(output)}
