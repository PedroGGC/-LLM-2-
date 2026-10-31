"""Cycle 5 uses the bounded two-lane engine with an independent protocol."""
from types import SimpleNamespace
from . import cycle4,cycle5_protocol as protocol,cycle5_report

STUDY=SimpleNamespace(cycle=5,protocol=protocol.PROTOCOL,protocol_id=protocol.PROTOCOL_ID,
    variants=protocol.VARIANTS,seeds=protocol.SEEDS,ordered_jobs=protocol.ordered_jobs,
    analyze=cycle5_report.analyze,post_seconds=1200,pilot_variant="fine_k4",post_all_seeds=False,
    depth_variants=protocol.VARIANTS[:-1],
    cpu_tests=["tests/test_cycle5.py","tests/test_campaign_storage.py","tests/test_json_portability.py"],
    cuda_tests="tests/test_cycle5.py")


def main(argv=None,*,started=None):
    parser=cycle4.parser();parser.set_defaults(export_policy="halfway-temporary")
    args=parser.parse_args(argv);args.study=STUDY
    if args.export_policy!="halfway-temporary":raise ValueError("Cycle 5 requires bounded archive retention")
    cycle4.validate(args)
    if not args.execute:return {"execution":"not_started","protocol":protocol.PROTOCOL,"protocol_id":protocol.PROTOCOL_ID}
    if not args.output:raise ValueError("fresh output directory required")
    return cycle4.execute(args,started=started,study=STUDY)

if __name__=="__main__":main()
