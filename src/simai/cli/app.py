import typer

from simai.cli.bench import app as bench_app
from simai.cli.generate import app as generate_app
from simai.cli.install import app as install_app
from simai.cli.profile import app as profile_app
from simai.cli.simulate import app as simulate_app

app = typer.Typer(
    name="simai",
    help="SimAI — AI datacenter network simulation toolkit.",
    no_args_is_help=True,
)

app.add_typer(bench_app, name="bench", help="Run distributed training benchmarks.")
app.add_typer(generate_app, name="generate", help="Generate workloads and topologies.")
app.add_typer(generate_app, name="gen", hidden=True)  # alias
app.add_typer(install_app, name="install", help="Build and install optional backends.")
app.add_typer(profile_app, name="profile", help="Profile GPU kernel execution times.")
app.add_typer(simulate_app, name="simulate", help="Run network simulations.")


def main():
    app()
