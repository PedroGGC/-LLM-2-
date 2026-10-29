"""Run or plan Cycle 1 (Context/State Balance) campaign.

Protocol:
- 5 variants:
    1. baseline_small (Transformer 0015)
    2. lrm_no_jepa_k2 (LRM K=2 atual)
    3. lrm_no_jepa_k2_rms (LRM K=2 com contexto normalizado)
    4. flrm_compressed_native_cached_no_jepa_k2 (FLRM N1 K=2 atual)
    5. flrm_compressed_native_cached_no_jepa_k2_rms (FLRM N1 K=2 com contexto normalizado)
- Seeds: 71, 72, 73
- Steps: 2000
- Effective batch: 32 (batch_size=32, grad_accum_steps=1)
- FP32 / Inductor (compile=True)
- Full evaluations: checkpoints nos passos 500, 1000, 1500, 2000
- Diagnósticos: diagnose_init=True, diagnose_steps=[500, 1000, 1500, 2000]
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm2.research.campaign import parser, run_campaign

CYCLE1_MODELS = [
    "baseline_small",
    "lrm_no_jepa_k2",
    "lrm_no_jepa_k2_rms",
    "flrm_compressed_native_cached_no_jepa_k2",
    "flrm_compressed_native_cached_no_jepa_k2_rms",
]
CYCLE1_SEEDS = [71, 72, 73]
CYCLE1_STEPS = 2000


def main(argv=None):
    p = parser()
    p.set_defaults(
        models=CYCLE1_MODELS,
        seeds=CYCLE1_SEEDS,
        steps=CYCLE1_STEPS,
        batch_size=32,
        grad_accum_steps=1,
        compile=True,
        paired_common_init=True,
        evaluate_all_checkpoints=True,
        diagnose_init=True,
        diagnose_steps=[500, 1000, 1500, 2000],
        diagnose_batch_size=4,
    )
    args = p.parse_args(argv)
    try:
        result = run_campaign(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"\n❌ PROCESSO INTERROMPIDO: {exc}\nConfira o motivo e os logs indicados acima.", flush=True)
        raise
    if result.get("status") == "complete":
        print("\n✓ PROCESSO CONCLUÍDO COM SUCESSO. O ZIP de análise está indicado acima.", flush=True)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
