# Contributing to SymSorter

Thank you for your interest in contributing to SymSorter! This document provides guidelines and information for contributors.

## Getting Started

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YourUsername/symsorter.git
   cd symsorter
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   pip install git+https://github.com/openai/CLIP.git
   ```

4. **Run tests to ensure everything works**:
   ```bash
   pytest
   ```

## Development Guidelines

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks:
```bash
black symsorter/
isort symsorter/
flake8 symsorter/
mypy symsorter/
```

### Testing

- Write tests for new functionality
- Ensure existing tests pass
- Aim for good test coverage

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=symsorter
```

### Documentation

- Update docstrings for new functions/classes
- Update README.md for user-facing changes
- Add examples for new features

## Types of Contributions

### Bug Reports

When filing a bug report, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior  
- Environment details (OS, Python version, etc.)
- Screenshots if applicable

### Feature Requests

For new features:
- Describe the use case
- Explain why it would be valuable
- Consider implementation complexity
- Discuss potential alternatives

### Code Contributions

1. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines above

3. **Test your changes** thoroughly

4. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: descriptive message"
   ```

5. **Push and create a pull request**

### Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add yourself to contributors if desired
4. Submit PR with clear description
5. Respond to review feedback

## Architecture Overview

### Core Components

- **`image_browser.py`**: Main GUI application class
- **`clip_encode.py`**: CLIP encoding functionality  
- **`cli.py`**: Command-line interface
- **`gui.py`**: GUI launcher

### Key Concepts

- **Smart Caching**: LRU cache for processed thumbnails and raw images
- **Multi-threading**: Background loading with worker threads
- **NPZ Storage**: Efficient storage of embeddings and metadata
- **Qt Integration**: Professional GUI with menus, toolbars, shortcuts

## Areas for Contribution

### High Priority
- Additional image format support
- Performance optimizations
- Cross-platform testing
- Documentation improvements

### Medium Priority  
- Additional similarity metrics
- Batch classification tools
- Integration with other ML frameworks
- Advanced filtering options

### Future Ideas
- Plugin system
- Web interface
- Integration with cloud storage
- Advanced visualization features

## Release Process

1. Update version in `pyproject.toml` and `__init__.py`
2. Update `CHANGELOG.md`
3. Create release tag
4. Build and upload to PyPI
5. Update documentation

## Questions and Support

- **General Questions**: Use GitHub Discussions
- **Bug Reports**: Create GitHub Issues  
- **Development Chat**: Contact maintainers directly

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Git commit history

Thank you for contributing to SymSorter!
