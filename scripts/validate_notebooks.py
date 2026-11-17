"""Execute all teaching notebooks in fresh kernels; save results outside source."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
output = root / "executed_notebooks"
output.mkdir(exist_ok=True)
for path in sorted((root / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    # CI dependencies are installed once; skip only the installation cell.
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.source.startswith("%pip"):
            cell.source = "# Dependencies installed before execution."
    nbformat.validate(notebook)
    NotebookClient(notebook, timeout=300, kernel_name="python3",
                   resources={"metadata": {"path": str(root)}}).execute()
    nbformat.write(notebook, output / path.name)
    print(f"PASS {path.name}", flush=True)
