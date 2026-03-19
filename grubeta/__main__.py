"""
Command-line interface for grubeta.

Usage:
    python -m grubeta AAPL SPY
    python -m grubeta AAPL SPY --start 2020-01-01 --preset responsive
    python -m grubeta AAPL MSFT GOOGL --market SPY --compare
    python -m grubeta AAPL SPY --report report.html
    python -m grubeta --list-presets
    grubeta AAPL SPY  (after pip install)
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="grubeta",
        description="Estimate time-varying CAPM beta using neural networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m grubeta AAPL SPY
  python -m grubeta AAPL SPY --start 2020-01-01 --preset responsive
  python -m grubeta AAPL MSFT GOOGL --market SPY --compare
  python -m grubeta AAPL SPY --report report.html
  python -m grubeta --list-presets
        """,
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Stock ticker(s). Last one is market unless --market is specified",
    )
    parser.add_argument(
        "--market", "-m", default=None, help="Market index ticker (default: last positional arg, or SPY)"
    )
    parser.add_argument("--start", "-s", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", "-e", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--preset",
        "-p",
        default="default",
        # NOTE: Keep in sync with grubeta/presets.py PRESETS dict
        choices=["default", "responsive", "smooth", "research"],
        help="Configuration preset (default: 'default')",
    )
    parser.add_argument("--compare", action="store_true", help="Compare multiple stocks' betas")
    parser.add_argument("--report", default=None, help="Generate HTML report to this file path")
    parser.add_argument("--no-plot", action="store_true", help="Suppress plot display")
    parser.add_argument("--output", "-o", default=None, help="Save beta time series to CSV")
    parser.add_argument("--list-presets", action="store_true", help="Show available configuration presets")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()

    if args.list_presets:
        from grubeta.presets import list_presets

        print("\nAvailable presets:\n")
        for name, desc in list_presets().items():
            print(f"  {name:12s}  {desc}")
        print()
        return

    if not args.tickers:
        parser.print_help()
        sys.exit(1)

    # Parse tickers
    if args.market:
        stocks = args.tickers
        market = args.market
    elif len(args.tickers) == 1:
        stocks = args.tickers
        market = "SPY"
    else:
        stocks = args.tickers[:-1]
        market = args.tickers[-1]

    from grubeta.convenience import compare_betas, estimate_beta

    if args.compare or len(stocks) > 1:
        result = compare_betas(
            stocks=stocks,
            market=market,
            start=args.start,
            end=args.end,
            preset=args.preset,
        )
    else:
        result = estimate_beta(
            stock=stocks[0],
            market=market,
            start=args.start,
            end=args.end,
            preset=args.preset,
            plot=not args.no_plot,
            verbose=not args.quiet,
        )

    # Print summary
    print(result["summary"])

    # Save CSV if requested
    if args.output:
        result["beta"].to_csv(args.output)
        print(f"\nBeta series saved to {args.output}")

    # Generate report if requested
    if args.report:
        from grubeta.convenience import quick_report

        quick_report(stock=stocks[0], market=market, output=args.report)
        print(f"\nReport saved to {args.report}")


if __name__ == "__main__":
    main()
