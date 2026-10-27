"""Run one experiment from the command line.

    python -m biotx.run experiments/annulus_biot2.yaml
    python -m biotx.run experiments/annulus_biot2.yaml --set discretization.deg=1 --set time.dt=0.005
    python -m biotx.run experiments/annulus_biot2.yaml --set geometry.h=0.00025 --name annulus_ref

--set takes dotted keys of the YAML tree; values are parsed as YAML (numbers, lists, strings).
"""
import argparse

import yaml

import biotx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--name", help="output file stem (default: output.name or the YAML stem)")
    args = ap.parse_args(argv)

    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    if args.name:
        overrides["output.name"] = args.name
    exp = biotx.load(args.yaml, overrides)

    biotx.simulate(exp)

if __name__ == "__main__":
    main()
