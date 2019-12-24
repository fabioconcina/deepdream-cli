#!/usr/bin/env bash
echo Running mypy...
python -m mypy .
echo ...END
echo Running pytest...
python -m pytest
echo ...END
echo Running flake8...
python -m flake8 .
echo ...END
