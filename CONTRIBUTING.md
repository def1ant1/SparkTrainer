# Contributing to SparkTrainer

Thank you for your interest in contributing to SparkTrainer! We welcome contributions from the community and are grateful for your support.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/SparkTrainer.git
   cd SparkTrainer
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/def1ant1/SparkTrainer.git
   ```
4. **Set up your development environment** following the [Developer Guide](DEVELOPER_GUIDE.md)

## How to Contribute

### Reporting Bugs

Before creating a bug report:
- Check the [existing issues](https://github.com/def1ant1/SparkTrainer/issues) to avoid duplicates
- Gather information about the bug (error messages, logs, environment details)

When filing a bug report, use the **Bug Report** template and include:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Environment details (OS, Python version, GPU, etc.)
- Relevant logs and error messages
- Screenshots if applicable

### Suggesting Features

We welcome feature suggestions! Before submitting:
- Check if the feature has already been requested
- Consider if it fits the project's scope and goals

Use the **Feature Request** template and include:
- A clear description of the problem this feature would solve
- Your proposed solution
- Alternative solutions you've considered
- Any relevant examples or mockups

### Contributing Code

We accept the following types of code contributions:

1. **Bug Fixes** - Fix reported issues
2. **New Features** - Add new functionality
3. **Recipe Contributions** - Add new training recipes
4. **Documentation** - Improve docs, examples, and tutorials
5. **Tests** - Increase test coverage
6. **Performance** - Optimize existing code

### Contributing Recipes

Training recipes are a core part of SparkTrainer. To contribute a new recipe:

1. Create your recipe following the `RecipeInterface` pattern
2. Add comprehensive documentation and docstrings
3. Include example usage
4. Add tests with >80% coverage
5. Submit via the **Recipe Submission** template

See [Developer Guide - Adding New Features](DEVELOPER_GUIDE.md#adding-new-features) for detailed instructions.

## Development Workflow

### 1. Create a Branch

Always create a new branch for your changes:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements

### 2. Make Your Changes

- Write clean, readable code following our [coding standards](#coding-standards)
- Add or update tests to cover your changes
- Update documentation as needed
- Ensure all tests pass locally

### 3. Commit Your Changes

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```bash
# Format
<type>(<scope>): <subject>

# Examples
feat(recipes): add new LoRA fine-tuning recipe for video models
fix(api): handle edge case in dataset validation
docs(readme): update installation instructions
test(ingestion): add tests for video wizard workflow
```

**Commit types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `test` - Adding or updating tests
- `refactor` - Code refactoring
- `perf` - Performance improvements
- `chore` - Maintenance tasks
- `ci` - CI/CD changes

**Good commit practices:**
- Keep commits focused and atomic
- Write clear, descriptive commit messages
- Reference issue numbers (e.g., "fixes #123")

### 4. Keep Your Branch Updated

Regularly sync with upstream:

```bash
git fetch upstream
git rebase upstream/main
```

Resolve any conflicts that arise.

### 5. Run Tests Locally

Before submitting, ensure all tests pass:

```bash
# Run backend tests
pytest tests/ -v --cov=src --cov=backend

# Run linting
flake8 backend/ src/
black --check backend/ src/
isort --check-only backend/ src/

# Run frontend tests
cd frontend
npm test
npm run lint
npm run type-check
```

### 6. Push Your Changes

```bash
git push origin feature/your-feature-name
```

### 7. Create a Pull Request

1. Go to your fork on GitHub
2. Click "Compare & pull request"
3. Fill out the PR template completely
4. Link related issues
5. Request review from maintainers

## Coding Standards

### Python

- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting (line length: 127)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Add type hints to function signatures
- Write comprehensive docstrings (Google style)

**Example:**

```python
from typing import List, Optional, Dict, Any


def process_dataset(
    dataset_path: str,
    output_dir: str,
    batch_size: int = 32,
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Process a dataset and generate outputs.

    Args:
        dataset_path: Path to the input dataset
        output_dir: Directory for output files
        batch_size: Number of samples per batch (default: 32)
        config: Optional configuration dictionary

    Returns:
        List of output file paths

    Raises:
        ValueError: If dataset_path doesn't exist
        IOError: If output_dir is not writable

    Example:
        >>> process_dataset("data/train.jsonl", "outputs/", batch_size=64)
        ['outputs/batch_1.pt', 'outputs/batch_2.pt']
    """
    # Implementation
    pass
```

### TypeScript/React

- Use functional components with TypeScript
- Follow [Airbnb React Style Guide](https://github.com/airbnb/javascript/tree/master/react)
- Use ESLint and Prettier
- Define proper TypeScript interfaces/types
- Use meaningful variable names

**Example:**

```typescript
import React from 'react';

interface ExperimentCardProps {
  experimentId: number;
  name: string;
  status: 'running' | 'completed' | 'failed';
  metrics?: Record<string, number>;
  onSelect?: (id: number) => void;
}

/**
 * Card component for displaying experiment information
 */
export const ExperimentCard: React.FC<ExperimentCardProps> = ({
  experimentId,
  name,
  status,
  metrics,
  onSelect,
}) => {
  const handleClick = () => {
    onSelect?.(experimentId);
  };

  return (
    <div className="experiment-card" onClick={handleClick}>
      {/* Component JSX */}
    </div>
  );
};
```

### Documentation

- Add docstrings to all public modules, classes, functions
- Use clear, concise language
- Include examples where helpful
- Update relevant documentation when changing functionality

## Testing Requirements

All contributions must include appropriate tests:

### Required Test Coverage

- **New features**: >80% code coverage
- **Bug fixes**: Test reproducing the bug + verification of fix
- **Refactoring**: Maintain or improve existing coverage

### Test Types

1. **Unit Tests** - Test individual functions/classes
   ```python
   def test_manifest_writer():
       """Test ManifestWriter creates valid JSONL files."""
       writer = ManifestWriter("test.jsonl")
       writer.add_entry({"id": 1, "text": "test"})
       writer.close()
       assert os.path.exists("test.jsonl")
   ```

2. **Integration Tests** - Test component interactions
   ```python
   def test_api_create_experiment(client):
       """Test experiment creation via API."""
       response = client.post('/api/experiments', json={...})
       assert response.status_code == 201
   ```

3. **End-to-End Tests** - Test full workflows
   ```python
   def test_training_pipeline():
       """Test complete training pipeline."""
       # Setup, execution, validation
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=backend --cov-report=html

# Run specific test file
pytest tests/test_recipes.py

# Run tests matching pattern
pytest -k "test_lora"
```

## Pull Request Process

### PR Requirements

Before submitting your PR, ensure:

- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] New code has tests with >80% coverage
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] PR template is filled out completely
- [ ] Branch is up-to-date with main

### PR Template

When creating a PR, include:

1. **Description** - What does this PR do?
2. **Motivation** - Why is this change needed?
3. **Related Issues** - Link to issues (e.g., "Closes #123")
4. **Type of Change** - Bug fix, feature, docs, etc.
5. **Testing** - How was this tested?
6. **Screenshots** - For UI changes
7. **Checklist** - Confirm all requirements met

### Review Process

1. **Automated Checks** - CI pipeline must pass
2. **Code Review** - At least one maintainer approval required
3. **Discussion** - Address reviewer feedback
4. **Approval** - Maintainer approves the PR
5. **Merge** - Maintainer merges to main

### After Merge

- Your contribution will be included in the next release
- You'll be added to the contributors list
- Consider helping review other PRs!

## Issue Guidelines

### Issue Templates

Use the appropriate template:

1. **Bug Report** - Report a bug or error
2. **Feature Request** - Suggest a new feature
3. **Recipe Submission** - Submit a new training recipe
4. **Question** - Ask a question (consider Discussions first)

### Issue Best Practices

- **Search first** - Check if the issue already exists
- **Be specific** - Provide clear, detailed information
- **One issue per report** - Don't combine multiple issues
- **Follow up** - Respond to questions from maintainers
- **Be respectful** - Follow the Code of Conduct

### Issue Labels

Issues are tagged with labels:

- `bug` - Something isn't working
- `feature` - New feature or request
- `documentation` - Documentation improvements
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `priority: high` - High priority
- `status: in-progress` - Being worked on

## Community

### Getting Help

- **GitHub Discussions** - Ask questions, share ideas
- **Issues** - Report bugs, request features
- **Developer Guide** - Technical documentation

### Staying Updated

- Watch the repository for notifications
- Check the [Roadmap](README.md#roadmap) for planned features
- Review recent PRs and issues

### Recognition

Contributors are recognized in:
- GitHub contributors list
- Release notes
- Project README

---

Thank you for contributing to SparkTrainer! Your efforts help make this project better for everyone.

## Quick Reference

### First-Time Contributor Checklist

- [ ] Read the Code of Conduct
- [ ] Fork the repository
- [ ] Set up development environment
- [ ] Find a "good first issue"
- [ ] Create a branch
- [ ] Make changes and add tests
- [ ] Run tests locally
- [ ] Submit a pull request
- [ ] Respond to review feedback

### Getting Started Commands

```bash
# Setup
git clone https://github.com/your-username/SparkTrainer.git
cd SparkTrainer
pip install -r requirements.txt
pip install -r backend/requirements.txt
pip install -e .

# Development
git checkout -b feature/my-feature
# Make changes
pytest tests/ -v
black backend/ src/
isort backend/ src/
flake8 backend/ src/

# Submit
git add .
git commit -m "feat: add my feature"
git push origin feature/my-feature
# Create PR on GitHub
```

### Need Help?

- **Questions?** Open a [Discussion](https://github.com/def1ant1/SparkTrainer/discussions)
- **Found a bug?** Open an [Issue](https://github.com/def1ant1/SparkTrainer/issues)
- **Have a suggestion?** Open a [Feature Request](https://github.com/def1ant1/SparkTrainer/issues/new?template=feature_request.md)
