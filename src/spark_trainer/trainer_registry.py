"""
Trainer Plugin Autodiscovery System for SparkTrainer.

This module automatically discovers and registers trainer implementations
from the trainers/ directory by scanning for @trainer_template decorators.
"""

import os
import re
import ast
import importlib
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainerMetadata:
    """Metadata for a discovered trainer"""
    name: str
    description: str
    module_path: str
    class_name: str
    model_types: List[str]
    tags: List[str] = field(default_factory=list)
    author: Optional[str] = None
    version: Optional[str] = None
    requirements: List[str] = field(default_factory=list)
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())


class TrainerRegistry:
    """
    Registry for automatically discovered trainer implementations.

    This class scans Python files in the trainers/ directory for classes
    decorated with @trainer_template and registers them for use.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the trainer registry.

        Args:
            base_dir: Base directory containing trainers/ folder
        """
        if base_dir is None:
            # Default to src/spark_trainer
            self.base_dir = Path(__file__).parent
        else:
            self.base_dir = Path(base_dir)

        self.trainers_dir = self.base_dir / "trainers"
        self._registry: Dict[str, TrainerMetadata] = {}
        self._classes: Dict[str, type] = {}

    def discover_trainers(self, search_paths: Optional[List[str]] = None) -> Dict[str, TrainerMetadata]:
        """
        Discover all trainers with @trainer_template decorator.

        Args:
            search_paths: Additional paths to search (optional)

        Returns:
            Dictionary of trainer name -> metadata
        """
        if search_paths is None:
            search_paths = [str(self.trainers_dir)]

        discovered = {}

        for search_path in search_paths:
            path = Path(search_path)
            if not path.exists():
                logger.warning(f"Trainer search path does not exist: {path}")
                continue

            # Find all Python files
            for py_file in path.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue

                try:
                    metadata_list = self._scan_file(py_file)
                    for metadata in metadata_list:
                        discovered[metadata.name] = metadata
                        logger.info(f"Discovered trainer: {metadata.name} in {py_file.name}")
                except Exception as e:
                    logger.error(f"Error scanning {py_file}: {e}")

        self._registry.update(discovered)
        return discovered

    def _scan_file(self, file_path: Path) -> List[TrainerMetadata]:
        """
        Scan a Python file for @trainer_template decorated classes.

        Args:
            file_path: Path to the Python file

        Returns:
            List of discovered trainer metadata
        """
        discovered = []

        with open(file_path, 'r') as f:
            content = f.read()

        # Parse the AST
        try:
            tree = ast.parse(content)
        except SyntaxError:
            logger.error(f"Syntax error in {file_path}")
            return discovered

        # Find classes with @trainer_template decorator
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                metadata = self._extract_metadata(node, file_path)
                if metadata:
                    discovered.append(metadata)

        return discovered

    def _extract_metadata(self, class_node: ast.ClassDef, file_path: Path) -> Optional[TrainerMetadata]:
        """
        Extract metadata from a class node with @trainer_template decorator.

        Args:
            class_node: AST ClassDef node
            file_path: Path to the file containing the class

        Returns:
            TrainerMetadata if decorator found, None otherwise
        """
        # Check if class has @trainer_template decorator
        has_decorator = False
        decorator_args = {}

        for decorator in class_node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "trainer_template":
                has_decorator = True
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name) and decorator.func.id == "trainer_template":
                    has_decorator = True
                    # Extract decorator arguments
                    for keyword in decorator.keywords:
                        try:
                            decorator_args[keyword.arg] = ast.literal_eval(keyword.value)
                        except:
                            pass

        if not has_decorator:
            return None

        # Extract docstring
        docstring = ast.get_docstring(class_node) or ""

        # Parse metadata from docstring or decorator args
        name = decorator_args.get('name', class_node.name)
        description = decorator_args.get('description', docstring.split('\n')[0] if docstring else "")
        model_types = decorator_args.get('model_types', [])
        tags = decorator_args.get('tags', [])
        author = decorator_args.get('author')
        version = decorator_args.get('version', '1.0.0')
        requirements = decorator_args.get('requirements', [])

        # If no metadata in decorator, try to extract from docstring
        if not model_types and docstring:
            model_types_match = re.search(r'Model Types?:\s*(.+)', docstring, re.IGNORECASE)
            if model_types_match:
                model_types = [mt.strip() for mt in model_types_match.group(1).split(',')]

        if not tags and docstring:
            tags_match = re.search(r'Tags:\s*(.+)', docstring, re.IGNORECASE)
            if tags_match:
                tags = [t.strip() for t in tags_match.group(1).split(',')]

        # Convert module path to importable format
        module_path = self._file_to_module_path(file_path)

        return TrainerMetadata(
            name=name,
            description=description,
            module_path=module_path,
            class_name=class_node.name,
            model_types=model_types,
            tags=tags,
            author=author,
            version=version,
            requirements=requirements
        )

    def _file_to_module_path(self, file_path: Path) -> str:
        """
        Convert a file path to a Python module path.

        Args:
            file_path: Path to Python file

        Returns:
            Module path string (e.g., 'spark_trainer.trainers.module')
        """
        # Get relative path from base_dir
        try:
            rel_path = file_path.relative_to(self.base_dir.parent)
        except ValueError:
            # If file is not under base_dir, use absolute path
            rel_path = file_path

        # Convert to module path
        parts = rel_path.parts[:-1] + (rel_path.stem,)
        return '.'.join(parts)

    def load_trainer(self, name: str) -> Optional[type]:
        """
        Load a trainer class by name.

        Args:
            name: Trainer name

        Returns:
            Trainer class, or None if not found
        """
        if name in self._classes:
            return self._classes[name]

        metadata = self._registry.get(name)
        if not metadata:
            logger.error(f"Trainer not found: {name}")
            return None

        try:
            # Import the module
            module = importlib.import_module(metadata.module_path)

            # Get the class
            trainer_class = getattr(module, metadata.class_name)

            # Cache it
            self._classes[name] = trainer_class

            return trainer_class
        except Exception as e:
            logger.error(f"Error loading trainer {name}: {e}")
            return None

    def get_trainer_metadata(self, name: str) -> Optional[TrainerMetadata]:
        """
        Get metadata for a trainer.

        Args:
            name: Trainer name

        Returns:
            TrainerMetadata or None
        """
        return self._registry.get(name)

    def list_trainers(self, model_type: Optional[str] = None,
                     tag: Optional[str] = None) -> List[TrainerMetadata]:
        """
        List all discovered trainers, optionally filtered.

        Args:
            model_type: Filter by model type
            tag: Filter by tag

        Returns:
            List of trainer metadata
        """
        trainers = list(self._registry.values())

        if model_type:
            trainers = [t for t in trainers if model_type in t.model_types]

        if tag:
            trainers = [t for t in trainers if tag in t.tags]

        return trainers

    def get_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the complete registry as a dictionary.

        Returns:
            Dictionary of trainer name -> metadata dict
        """
        return {
            name: {
                'name': meta.name,
                'description': meta.description,
                'module_path': meta.module_path,
                'class_name': meta.class_name,
                'model_types': meta.model_types,
                'tags': meta.tags,
                'author': meta.author,
                'version': meta.version,
                'requirements': meta.requirements,
                'discovered_at': meta.discovered_at
            }
            for name, meta in self._registry.items()
        }


# Decorator for marking trainer classes
def trainer_template(**kwargs):
    """
    Decorator for marking trainer classes for autodiscovery.

    Usage:
        @trainer_template(
            name="CustomTrainer",
            description="My custom trainer",
            model_types=["vision_language"],
            tags=["custom", "experimental"],
            author="Your Name",
            version="1.0.0",
            requirements=["transformers>=4.30.0"]
        )
        class CustomTrainer:
            ...

    Args:
        name: Trainer name (defaults to class name)
        description: Brief description
        model_types: List of supported model types
        tags: List of tags for categorization
        author: Author name
        version: Version string
        requirements: List of Python package requirements
    """
    def decorator(cls):
        # Store metadata as class attributes
        cls._trainer_template_metadata = kwargs
        return cls
    return decorator


# Global registry instance
_global_registry: Optional[TrainerRegistry] = None


def get_trainer_registry(base_dir: Optional[Path] = None) -> TrainerRegistry:
    """
    Get or create the global trainer registry instance.

    Args:
        base_dir: Base directory (optional, uses default if None)

    Returns:
        TrainerRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = TrainerRegistry(base_dir)
        # Auto-discover trainers on first access
        _global_registry.discover_trainers()
    return _global_registry


def reload_registry(base_dir: Optional[Path] = None) -> TrainerRegistry:
    """
    Reload the trainer registry (useful for development).

    Args:
        base_dir: Base directory (optional)

    Returns:
        New TrainerRegistry instance
    """
    global _global_registry
    _global_registry = TrainerRegistry(base_dir)
    _global_registry.discover_trainers()
    return _global_registry
