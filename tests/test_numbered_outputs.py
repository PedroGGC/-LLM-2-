"""Output naming must never reuse or overwrite previous experiment artifacts."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import runpy

import pytest

from llm2.research.campaign import Campaign, export_campaign
from llm2.research.naming import numbered_output


def test_sequential_reservations_allow_exclusive_campaign_creation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    first = numbered_output(tmp_path)
    assert first.name == "llm2-campaign-0001" and not first.exists()
    campaign = Campaign(root, first, device="cpu", gpus=[])
    archive = export_campaign(campaign.output)
    assert archive.name == "llm2-campaign-0001-results.zip"
    second = numbered_output(tmp_path)
    assert second.name == "llm2-campaign-0002" and not second.exists()
    # A failed bootstrap's reservation is never reused in the same workspace.
    assert numbered_output(tmp_path).name == "llm2-campaign-0003"
    assert archive.is_file() and (first / "campaign.json").is_file()


def test_observes_zip_only_input_only_and_previous_attached_outputs(tmp_path):
    work, inputs = tmp_path / "working", tmp_path / "input"
    work.mkdir()
    old_zip = work / "llm2-campaign-0007-results.zip"
    old_zip.write_bytes(b"do not overwrite")
    (work / "llm2-input-0008").mkdir()
    (work / "llm2-campaign-a668af73fde0").mkdir()  # Historical hash is ignored.
    mounted = inputs / "old-notebook" / "outputs"
    mounted.mkdir(parents=True)
    (mounted / "llm2-campaign-0012").mkdir()
    assert numbered_output(work, search_roots=[inputs]).name == "llm2-campaign-0013"
    assert old_zip.read_bytes() == b"do not overwrite"
    with pytest.raises(FileExistsError, match="já foi usado"):
        numbered_output(work, number=12, search_roots=[inputs])


def test_manual_number_and_review_sequence_are_independent(tmp_path):
    assert numbered_output(tmp_path, number=25).name == "llm2-campaign-0025"
    with pytest.raises(FileExistsError):
        numbered_output(tmp_path, number=25)
    assert numbered_output(tmp_path).name == "llm2-campaign-0026"
    assert numbered_output(tmp_path, "llm2-review").name == "llm2-review-0001"
    assert numbered_output(tmp_path, "llm2-review").name == "llm2-review-0002"


@pytest.mark.parametrize("value", [0, -1, True, "1", 1.5])
def test_invalid_manual_number_cannot_create_output(tmp_path, value):
    parent = tmp_path / "new"
    with pytest.raises(ValueError, match="inteiro positivo"):
        numbered_output(parent, number=value)
    assert not parent.exists()


def test_concurrent_reservations_are_unique_and_counter_continues(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda _: numbered_output(tmp_path), range(24)))
    assert len(set(paths)) == 24
    assert {p.name for p in paths} == {f"llm2-campaign-{n:04d}" for n in range(1, 25)}
    assert not any(p.exists() for p in paths)
    assert numbered_output(tmp_path).name == "llm2-campaign-0025"


def test_bootstrap_can_load_pinned_helper_without_importing_package(tmp_path):
    path = Path(__file__).resolve().parents[1] / "src/llm2/research/naming.py"
    helper = runpy.run_path(str(path))["numbered_output"]
    assert helper(tmp_path).name == "llm2-campaign-0001"
