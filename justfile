default: testfast

# Install the virtual environment and install the pre-commit hooks
install:
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

# Run code quality tools.
check:
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uv run deptry src

# Test the code with pytest
test:
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --tb=short --cov --cov-config=pyproject.toml --cov-report=html

# Test the code with pytest
testfast:
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest -m "not slow" --tb=short --cov --cov-config=pyproject.toml --cov-report=html

# Build wheel file
build: clean-build
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

# Clean build artifacts
clean-build:
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

# Publish a release to PyPI.
publish:
	@echo "🚀 Publishing."
	@uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

# Build and publish.
build-and-publish: build publish

# Test if documentation can be built without warnings or errors
docs-test:
	@uv run mkdocs build -s

# Build and serve the documentation
docs:
	@uv run mkdocs serve

edit:
  @uv run vim

e: edit
vim: edit

repomix:
  repomix --style plain src tests
