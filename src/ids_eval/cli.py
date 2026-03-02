from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ids_eval.version import __version__

app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    help="""
    IDS Evaluation Framework: A comprehensive, modular, and configurable framework for
    evaluating Machine Learning-based Intrusion Detection Systems.
    """,
    rich_markup_mode="markdown",
)
console = Console()


@app.callback()
def callback() -> None:
    pass


@app.command()
def version() -> None:
    console.print(f"IDS-EVAL Version: [bold green]{__version__}[/bold green]")


@app.command()
def dataset(
    config_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to configuration YAML file.", file_okay=True, dir_okay=False
    )
) -> None:
    console.print(f":arrow_forward: Starting data ingestion and preparation using config: '{config_path}'")
    from ids_eval.dataset_pipeline.data_manager import DataManager
    from ids_eval.run_config_pipeline.config_manager import ConfigManager

    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.load()
        pipeline = DataManager(config)
        pipeline.run()
        config_manager.store_config(config)
        console.print(":arrow_forward: [bold green]Dataset pipeline completed successfully![/bold green]")
    except Exception as e:
        console.print(f":x: [bold red]An error occurred in dataset pipeline:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    config_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to configuration YAML file.", file_okay=True, dir_okay=False
    ),
    train_only: bool = typer.Option(False, "--train-only", help="Only perform training, skip testing phase."),
    force_train: bool = typer.Option(
        False, "--force-train", help="Ignore saved models and force retraining (overwrites existing saved models)."
    ),
    force_model: bool = typer.Option(
        False,
        "--force-model",
        help="Force loading saved models without config hash validation. "
        "Allows evaluating models trained with a different configuration.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Clear all evaluation checkpoints before running. "
        "Forces a fresh evaluation run without retraining cached models, if any.",
    ),
) -> None:
    mode_parts = []
    if train_only:
        mode_parts.append("train-only")
    if force_train:
        mode_parts.append("force-train")
    if force_model:
        mode_parts.append("force-model")
    if clear_checkpoints:
        mode_parts.append("clear-checkpoints")
    mode_info = f" ({', '.join(mode_parts)})" if mode_parts else ""
    console.print(f":arrow_forward: Starting IDS evaluation{mode_info} using config: '{config_path}'")
    from ids_eval.evaluation_pipeline.evaluation_manager import EvaluationManager
    from ids_eval.run_config_pipeline.config_manager import ConfigManager

    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.load()
        pipeline = EvaluationManager(
            config,
            train_only=train_only,
            force_train=force_train,
            force_model=force_model,
            clear_checkpoints=clear_checkpoints,
        )
        pipeline.run()
        config_manager.store_config(config)
        console.print(":arrow_forward: [bold green]Evaluation completed successfully![/bold green]")
    except Exception as e:
        console.print(f":x: [bold red]An error occurred in evaluation pipeline:[/bold red] {e}")
        raise typer.Exit(code=1)
