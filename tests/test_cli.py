import json
import os
import subprocess
import sys

import yaml

from llm2.cli import main
from test_training import training_config


def test_cli_end_to_end(training_config, tmp_path, capsys):
    cfg = training_config
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    main(["inspect", "--config", str(config)])
    result = json.loads(capsys.readouterr().out)
    assert result["parameters_calculated"] == result["parameters_formula"]
    main(["train", "--config", str(config), "--until-step", "2"])
    checkpoint = json.loads(capsys.readouterr().out)["checkpoint"]
    main(["evaluate", "--checkpoint", checkpoint, "--max-batches", "1"])
    assert json.loads(capsys.readouterr().out)["validation_tokens"] == 16
    main(["generate", "--checkpoint", checkpoint, "--prompt", "Science", "--max-new-tokens", "4"])
    assert capsys.readouterr().out.startswith("Science")
    # Regression: legacy Windows/ASCII pipes must print arbitrary generated Unicode.
    process = subprocess.run(
        [sys.executable, "-m", "llm2", "generate", "--checkpoint", checkpoint,
         "--prompt", "ciência 漢字", "--max-new-tokens", "1", "--temperature", "0"],
        env={**os.environ, "PYTHONIOENCODING": "ascii"}, capture_output=True, check=True,
    )
    assert process.stdout.decode("utf-8").startswith("ciência 漢字")
    main(["train", "--config", str(config), "--resume", checkpoint, "--max-steps", "4"])
    result = json.loads(capsys.readouterr().out)
    assert result["checkpoint"].endswith("step-00000004")
    assert result["tokens_processed"] == 4 * 2 * 2 * 8

