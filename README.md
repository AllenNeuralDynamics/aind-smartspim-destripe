# aind-smartspim-destripe

Source code to remove streaks from lightsheet images acquired with the SmartSPIM microscope. Currently, we are using a log spatial Fast Fourier Transform to remove the streaks. This works for us since our brains have cells with high intensity values that when we apply a dual-band filtering, artifacts are generated. Currently, this version works with Zarr data.

![raw data](https://github.com/AllenNeuralDynamics/aind-smartspim-destripe/blob/main/metadata/imgs/raw.png?raw=true)

## Dual-band
![dual band filtering](https://github.com/AllenNeuralDynamics/aind-smartspim-destripe/blob/main/metadata/imgs/filtered_dual_band.png?raw=true)

## Log Space FFT
![log space filtering](https://github.com/AllenNeuralDynamics/aind-smartspim-destripe/blob/main/metadata/imgs/filtered_log_space.png?raw=true)

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e .[dev]
```

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```bash
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```bash
black .
```

- Use **isort** to automatically sort import statements:
```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o doc_template/source/ src 
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html doc_template/source/ doc_template/build/html
```
More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).