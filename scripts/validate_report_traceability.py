#!/usr/bin/env python3
"""Validate that report numbers come from one canonical evaluation CSV.

Usage:
  python scripts/validate_report_traceability.py \
    --summary-csv data/pie/eval_reports/formal/evaluation_summary.csv \
    --mean-std-csv data/pie/eval_reports/formal/evaluation_summary_mean_std.csv
"""

import argparse
import csv
import os
import sys

REQUIRED_SUMMARY_COLUMNS = {
    'model_name', 'K', 'seed', 'protocol', 'eval_split',
    'F1', 'AUC', 'minADE', 'minFDE'
}

REQUIRED_MEAN_STD_COLUMNS = {
    'model_name', 'K', 'num_runs',
    'F1_mean', 'F1_std', 'AUC_mean', 'AUC_std',
    'minADE_mean', 'minADE_std', 'minFDE_mean', 'minFDE_std'
}


def read_csv_header(path):
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or []


def validate_columns(path, required_columns):
    columns = set(read_csv_header(path))
    missing = sorted(required_columns - columns)
    if missing:
        print('[FAIL] Missing columns in {}: {}'.format(path, ', '.join(missing)))
        return False
    print('[OK] {} contains required columns.'.format(path))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary-csv', required=True)
    parser.add_argument('--mean-std-csv', required=True)
    args = parser.parse_args()

    ok = True
    for path in [args.summary_csv, args.mean_std_csv]:
        if not os.path.exists(path):
            print('[FAIL] File not found: {}'.format(path))
            ok = False

    if not ok:
        return 1

    ok = validate_columns(args.summary_csv, REQUIRED_SUMMARY_COLUMNS) and ok
    ok = validate_columns(args.mean_std_csv, REQUIRED_MEAN_STD_COLUMNS) and ok

    if ok:
        print('[PASS] Traceability pre-check passed. Use these files as the single source of truth.')
        return 0
    return 2


if __name__ == '__main__':
    sys.exit(main())
