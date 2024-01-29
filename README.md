
# Project Title

This repository contains the codebase for [MLOPsPractical]. It utilizes [Poetry](https://python-poetry.org/) for package management, ensuring a clean and reproducible environment. Follow the steps below to set up the project on your local machine.

## Installation

### 1. Install Poetry

Poetry is a dependency manager for Python projects. If you don't have Poetry installed, you can do so by following the instructions on the official website: [Poetry Installation Guide](https://python-poetry.org/docs/#installation)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Add Necessary Packages

Once Poetry is installed, navigate to the project directory and run the following command to add the required packages:

```bash
poetry add scikit-learn pandas flask
```

This command installs the specified packages (scikit-learn, pandas, flask) and their dependencies, creating a `pyproject.toml` file to manage the project's dependencies.

### 3. Optional: Install Additional Packages

You can use the same `poetry add` command to add any other required packages. For example, if you need to use pickle:

```bash
poetry add pickle
```

### 4. Activate the Virtual Environment

Poetry automatically creates a virtual environment for your project. Activate it with:

```bash
poetry shell
```

Now you're ready to start working on your project with the specified dependencies installed!



This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.
