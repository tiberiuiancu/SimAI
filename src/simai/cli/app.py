import typer

from simai.cli.generate import app as generate_app
from simai.cli.simulate import app as simulate_app

app = typer.Typer(
    name="simai",
    help="SimAI — AI datacenter network simulation toolkit.",
    no_args_is_help=True,
)

app.add_typer(generate_app, name="generate", help="Generate workloads and topologies.")
app.add_typer(generate_app, name="gen", hidden=True)  # alias
app.add_typer(simulate_app, name="simulate", help="Run network simulations.")


def main():
    app()
