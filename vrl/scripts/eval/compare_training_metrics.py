"""Compare an explicit exact metric protocol across completed supervised runs."""

import argparse
import json
from pathlib import Path

from vrl.trainers.evidence import compare_run_metrics, publish_evidence_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-receipt", type=Path, required=True)
    parser.add_argument("--reference-verdict", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--candidate-verdict", type=Path, required=True)
    parser.add_argument("--columns", nargs="+", required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError(args.report)
    result = compare_run_metrics(
        args.reference_receipt,
        args.reference_verdict,
        args.candidate_receipt,
        args.candidate_verdict,
        columns=tuple(args.columns),
        expected_epochs=args.expected_epochs,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    publish_evidence_record(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
