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

### Immediate (Phase 0)
- [ ] Implement token-aware `build_context(max_tokens=N)`
- [ ] Implement cascading context propagation (decisions bubble up)
- [ ] Add conflict detection for cross-session contradictions
- [ ] Add importance tiers (critical/normal/low) for recorded items

### Short-term (Phase 1)
- [ ] Complete SQLite storage backend (methods partially stubbed)
- [ ] Complete Redis storage backend (methods partially stubbed)
- [ ] Add persistent embedding cache
- [ ] Add thread-safe context management

### Infrastructure
- [ ] Update email in pyproject.toml
- [ ] Create GitHub repository
- [ ] Set up CI/CD (GitHub Actions)
- [ ] Set up ReadTheDocs
- [ ] Add more examples

### Long-term
- [ ] MCP server integration
- [ ] Memory consolidation & forgetting
- [ ] Multi-agent shared memory
- [ ] Async API support

See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md#roadmap-features-detailed-implementation) for detailed designs.
