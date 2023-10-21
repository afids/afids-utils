# Anatomical Fiducials (AFIDs) Utility Tools

[![Documentation Status](https://readthedocs.org/projects/afids-utils/badge/?version=stable)](https://afids-utils.readthedocs.io/en/stable/?badge=stable)

The `afids-utils` package provides common utilities for projects involving
anatomical fiducials (AFIDs). For a comprehensive list of available utilities
refer to the [documentation] page.

## Installation

`afids-utils` can be installed using pip:

```bash
pip install afids-utils
```

## Contributing

`afids-utils` is an open-source project and contributions are welcome! If you
have any bug reports, feature requests, or improvement suggestions, please
submit them to the [issues page](https://github.com/afids/afids-utils/issues).

To contribute, first click the "fork" button to create your own copy of the
repository and then clone the project to your local machine:

```bash
git clone https://github.com/your-username/afids-utils.git
```

Navigate to the location where the directory was cloned and add the upstream
repository:

```bash
cd afids-utils

git remote add upstream https://github.com/afids/afids-utils.git
```

Now, `git remote -v` will show two remote repositories:

- `upstream`, referring to the `afids` repository
- `origin`, your personal fork

To the pull the latest changes from the code:

```bash
git checkout main
git pull upstream main
```

To develop and push your contribution to your copy of the repository:

```bash
git checkout -b contribution-name
git push origin contribution-name
```

Once pushed, a pull request can be opened, at which point the contribution
will be reviewed by a maintainer of the `afids-utils` repository.

### Poetry

`afids-utils` depedencies are managed with Poetry (version 1.2.0+). Please
refer to the [poetry website] for installation instructions. Following the
installation of Poetry, the development environment can be set up from the
local repository location by running the following commands:

```bash
poetry install --with dev
```

Poetry uses [poethepoet] as a task runner. You can see what commands are
available by running:

```bash
poetry run poe
```

Tests are performed with `pytest` and can be run via:

```bash
poetry run poe test
```

Additionally, `afids-utils` makes use of a number of libraries to lint and
format code, which can be invoked by running the following:

```bash
poetry run poe setup
```

This sets up a pre-commit hook, which runs the ncessary checks every time
a commit is performed. Alternatively, if you don't wish to use the pre-commit
hook, you can run the following manually, which performs the same checks:

```bash
poetry run poe quality
```

Please ensure these all pass before making a pull request.

[documentation]: https://afids-utils.readthedocs.io/en/stable
[issues page]: https://github.com/afids/afids-utils/issues
[poetry website]: https://python-poetry.org/docs/master/#installation
[poethepoet]: https://github.com/nat-n/poethepoet
