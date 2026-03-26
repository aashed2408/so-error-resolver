"""SO Error Resolver — CLI entry point.

Paste an error message and the tool will:
  1. Scrape Stack Overflow for relevant threads.
  2. Feed scraped data to an Ollama cloud/local model.
  3. If confidence is Low, refine the search and repeat until a proper fix
     is found or the maximum iteration limit is reached.

Default model: ``minimax/m2:5`` (Ollama cloud).
"""

from __future__ import annotations

import argparse
import asyncio
import os

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ai_engine import AIEngine, AIResolution
from proxy_manager import ProxyRotator
from scraper import StackOverflowScraper

console = Console()

BANNER = r"""
 ____  ___  ____  _____ ____  _   _ _____ _____ _   _ ____
/ ___|/ _ \|  _ \| ____/ ___|| | | | ____|_   _| | | |  _ \
| |  | | | | |_) |  _|| |  _ | |___|  _|   | | | | | | |_) |
| |__| |_| |  _ <| |__| |_| || |___| |___  | | | |_| |  __/
 \____\___/|_| \_\_____\____/ \____|_____| |_|  \___/|_|
"""

MAX_ITERATIONS = 5


def _get_multiline_input() -> str:
    """Read multi-line error input from the user (terminated by empty line)."""
    console.print(
        "\n[dim]Paste your error / traceback below. "
        "Press Enter on an empty line to submit.[/dim]\n"
    )
    lines: list[str] = []
    while True:
        try:
            line = console.input()
        except EOFError:
            break
        if line.strip() == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines)


def _display_result(resolution: AIResolution, iteration: int) -> None:
    """Pretty-print the AI resolution using rich panels and markdown."""
    console.print()
    console.print(Rule(f"Analysis Result (iteration {iteration})", style="blue"))
    console.print()

    # Root Cause
    console.print(
        Panel(
            Text(resolution.root_cause, style="bold white"),
            title="[bold red]Root Cause[/bold red]",
            border_style="red",
            padding=(1, 2),
        )
    )

    # Confidence
    color_map = {"High": "green", "Medium": "yellow", "Low": "red"}
    c_color = color_map.get(resolution.confidence, "white")
    console.print(
        f"\n  Confidence: [{c_color} bold]{resolution.confidence}[/{c_color} bold]"
    )

    # Fix Recommendation
    console.print()
    console.print(
        Panel(
            Markdown(resolution.fix_recommendation),
            title="[bold green]Fix Recommendation[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # Relevant Threads
    if resolution.relevant_threads:
        table = Table(
            title="Relevant Stack Overflow Threads",
            show_lines=False,
            border_style="blue",
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("URL", style="cyan", no_wrap=False)
        for idx, url in enumerate(resolution.relevant_threads, 1):
            table.add_row(str(idx), url)
        console.print()
        console.print(table)

    # Refined queries (if any)
    if resolution.refined_queries:
        console.print()
        console.print("  [dim]Refined search queries for next iteration:[/dim]")
        for rq in resolution.refined_queries:
            console.print(f"    [italic]{rq}[/italic]")

    console.print()


async def _run(
    error_text: str,
    *,
    model: str,
    host: str,
    no_proxy: bool,
    max_iterations: int,
) -> None:
    """Main async pipeline: iterative scrape → analyze → refine → repeat."""
    # ── Components ──
    proxy_rotator = (
        ProxyRotator(proxies=[]) if no_proxy else ProxyRotator()
    )
    scraper = StackOverflowScraper(proxy_rotator, max_retries=3)
    ai = AIEngine(model=model, host=host)

    # ── Ollama connectivity check ──
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Connecting to Ollama..."),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("check", total=None)
        connected, msg = await ai.check_connection()

    if not connected:
        console.print(
            Panel(
                f"{msg}\n\n"
                f"Model: [bold]{model}[/bold]\n"
                f"Host:  [bold]{host}[/bold]\n\n"
                "Troubleshooting:\n"
                "  1. Install Ollama: https://ollama.ai\n"
                "  2. Start the server: [dim]ollama serve[/dim]\n"
                f"  3. Pull the model:  [dim]ollama pull {model}[/dim]\n"
                "  4. For cloud models, ensure internet connectivity.",
                title="[bold red]Ollama Connection Failed[/bold red]",
                border_style="red",
            )
        )
        raise SystemExit(1)

    console.print(f"  [green]Ollama connected.[/green]  Model: [bold]{model}[/bold]\n")

    # ── Iterative scrape → analyze loop ──
    resolution: AIResolution | None = None
    previous_queries: list[str] = []
    new_questions_batch: list = []

    for iteration in range(1, max_iterations + 1):
        # Determine refined queries for this iteration
        refined = resolution.refined_queries if resolution else None
        if refined:
            previous_queries.extend(refined)

        # ── Scrape ──
        with Progress(
            SpinnerColumn(),
            TextColumn(
                f"[bold blue]Scraping Stack Overflow (iteration {iteration}/{max_iterations})..."
            ),
            transient=True,
            console=console,
        ) as progress:
            progress.add_task("scrape", total=None)
            new_questions_batch = await scraper.search_and_extract(
                error_text,
                max_results=5,
                refined_queries=refined,
            )

        total = scraper.total_scraped
        if new_questions_batch:
            console.print(
                f"  Found [green bold]{len(new_questions_batch)}[/green bold] new thread(s). "
                f"([dim]{total} total scraped[/dim])\n"
            )
        elif iteration == 1:
            console.print(
                "  [yellow]No threads found. "
                "Proceeding with traceback analysis only.[/yellow]\n"
            )
        else:
            console.print(
                "  [yellow]No new threads found for refined queries. "
                "Analyzing with existing data.[/yellow]\n"
            )

        # ── Analyze ──
        all_questions = scraper.get_all_questions()
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold magenta]Analyzing with {model}..."),
            transient=True,
            console=console,
        ) as progress:
            progress.add_task("analyze", total=None)
            resolution = await ai.analyze(
                error_text,
                all_questions,
                iteration=iteration,
                max_iterations=max_iterations,
                previous_queries=previous_queries,
            )

        _display_result(resolution, iteration)

        # ── Check if we have a proper fix ──
        if resolution.confidence == "High" and not resolution.needs_more_data:
            console.print(
                Panel(
                    f"High-confidence fix found after [bold]{iteration}[/bold] iteration(s) "
                    f"across [bold]{total}[/bold] Stack Overflow threads.",
                    title="[bold green]Done[/bold green]",
                    border_style="green",
                )
            )
            return

        if resolution.confidence == "Medium" and not resolution.needs_more_data:
            console.print(
                Panel(
                    f"Medium-confidence fix found after [bold]{iteration}[/bold] iteration(s). "
                    "Stopping early (model is satisfied with the data).",
                    title="[bold yellow]Done (Medium Confidence)[/bold yellow]",
                    border_style="yellow",
                )
            )
            return

        # ── Prepare for next iteration ──
        if iteration < max_iterations:
            if resolution.refined_queries:
                console.print(
                    f"\n  [dim]Confidence is {resolution.confidence}. "
                    "Refining search and retrying...[/dim]\n"
                )
            else:
                # AI didn't provide queries — generate fallback ones
                fallback = _generate_fallback_queries(error_text, previous_queries)
                resolution.refined_queries = fallback
                console.print(
                    f"\n  [dim]Confidence is {resolution.confidence}. "
                    "Generating fallback search queries...[/dim]\n"
                )

    # Max iterations reached
    console.print(
        Panel(
            f"Reached maximum iterations ([bold]{max_iterations}[/bold]) with "
            f"[bold]{scraper.total_scraped}[/bold] threads scraped.\n\n"
            "The best available analysis is shown above. Consider:\n"
            "  - Providing a more specific error message\n"
            "  - Using a more capable model\n"
            "  - Increasing --max-iterations",
            title="[bold yellow]Max Iterations Reached[/bold yellow]",
            border_style="yellow",
        )
    )


def _generate_fallback_queries(error_text: str, already_tried: list[str]) -> list[str]:
    """Generate fallback search queries when the AI doesn't provide them."""
    queries: list[str] = []
    lines = error_text.strip().splitlines()

    # Try to extract the exception class name
    for line in reversed(lines):
        line = line.strip()
        # Match patterns like "ExceptionName: message" or "module.ExceptionName:"
        match = re.search(r"([A-Z]\w+(?:Error|Exception|Warning))", line)
        if match:
            exc_name = match.group(1)
            queries.append(f"{exc_name} fix solution")
            queries.append(f"{exc_name} python example")
            break

    # Try the last non-empty line as a query
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("File ") and not line.startswith("^"):
            queries.append(line[:100])
            break

    # Filter out already-tried queries
    return [q for q in queries if q not in already_tried][:3]


import re  # noqa: E402 — needed by _generate_fallback_queries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="so-error-resolver",
        description=(
            "Resolve programming errors by iteratively scraping Stack Overflow "
            "and analyzing results with Ollama AI models (cloud or local)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                          # Interactive mode\n"
            "  %(prog)s -e 'TypeError: ...'       # Direct error input\n"
            "  %(prog)s -m llama3 --no-proxy      # Use local llama3, no proxy\n"
            "  %(prog)s -m minimax/m2:5 --max-iterations 8  # More iterations\n"
        ),
    )
    parser.add_argument(
        "-m", "--model",
        default=os.getenv("OLLAMA_MODEL", "minimax/m2:5"),
        help=(
            "Ollama model identifier. Supports cloud models like "
            "'minimax/m2:5' and local models like 'llama3' or 'mistral'. "
            "(default: minimax/m2:5, env: OLLAMA_MODEL)"
        ),
    )
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama API endpoint. (default: http://localhost:11434, env: OLLAMA_HOST)",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Skip proxy rotation — use direct connections.",
    )
    parser.add_argument(
        "-e", "--error",
        default=None,
        help="Error message / traceback. If omitted, read from stdin interactively.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=int(os.getenv("MAX_ITERATIONS", MAX_ITERATIONS)),
        help=(
            f"Maximum scrape-analyze-refine iterations. "
            f"(default: {MAX_ITERATIONS}, env: MAX_ITERATIONS)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    console.print(Text(BANNER, style="bold blue"))
    console.print(
        "  [dim]Stack Overflow Error Resolver — "
        "iterative scrape + AI analysis.[/dim]\n"
    )

    if args.error:
        error_text = args.error
    else:
        error_text = _get_multiline_input()

    if not error_text.strip():
        console.print("[red]No error provided. Exiting.[/red]")
        raise SystemExit(0)

    try:
        asyncio.run(
            _run(
                error_text,
                model=args.model,
                host=args.host,
                no_proxy=args.no_proxy,
                max_iterations=args.max_iterations,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
