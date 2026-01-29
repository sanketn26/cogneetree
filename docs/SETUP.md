# cogneetree Setup Instructions

## Development Setup

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Run tests**
   ```bash
   pytest
   ```

4. **Format code**
   ```bash
   black src tests
   isort src tests
   ```

## Publishing to PyPI

1. **Build package**
   ```bash
   python -m build
   ```

2. **Upload to TestPyPI (test first)**
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

3. **Upload to PyPI**
   ```bash
   python -m twine upload dist/*
   ```

## Next Steps

- [ ] Update email in pyproject.toml
- [ ] Create GitHub repository
- [ ] Set up ReadTheDocs
- [ ] Add more examples
- [ ] Implement storage backends (SQLite, Redis, PostgreSQL)
- [ ] Add comprehensive documentation
