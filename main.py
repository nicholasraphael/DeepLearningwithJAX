import typer

app = typer.Typer()


@app.command()
def inference(name: str):
    """
    Run inference on a given implementation.
    """
    print(f"Inference {name}")


@app.command()
def train(name: str):
    """
    Train a given implementation.
    """
    print(f"Training {name}")


if __name__ == "__main__":
    app()
