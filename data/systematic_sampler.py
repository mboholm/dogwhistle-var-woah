from all_spraak import tmp2simple
import argparse

def main(args):
	tmp2simple(
        path      = args.input, 
        out_dir   = args.output, 
        intervall = args.interval, 
        y0        = args.genesis
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="systematic_sampler.py", description="Systematic sampler of data from 'raw' Språkbanken JSON.")
    parser.add_argument("input", type=str, help="Path to input directory.")
    parser.add_argument("output", type=str, help="Path to output directory.")
    parser.add_argument("-n", "--interval", type=str, choices=["yearly", "semiannually", "quarterly", "monthly"], default="monthly", help="Provide to define intervall. Default = monthly.")
    parser.add_argument("-g", "--genesis", type=int, default=2000, help="To define the first year (y0). Default: 2000.")

    args = parser.parse_args()

    for k, v in args.__dict__.items(): 
        print(f"{k}: {v}")

    main(args)
