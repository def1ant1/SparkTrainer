"""
lakeFS Integration for Dataset Versioning.

Provides:
- Dataset version control (Git-like for data)
- Branch/commit/merge operations
- Data lineage tracking
- S3-compatible interface
- Integration with existing storage backends
"""

import os
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class LakeFSCommit:
    """lakeFS commit information."""
    id: str
    message: str
    committer: str
    creation_date: str
    metadata: Dict[str, Any]
    parents: List[str]


@dataclass
class LakeFSBranch:
    """lakeFS branch information."""
    name: str
    commit_id: str
    created_at: str


@dataclass
class LakeFSDatasetVersion:
    """Dataset version metadata."""
    repository: str
    branch: str
    commit_id: str
    path: str
    size_bytes: int
    num_files: int
    created_at: str
    metadata: Dict[str, Any]


class LakeFSClient:
    """
    lakeFS client for dataset version control.

    lakeFS provides Git-like version control for data lakes.
    """

    def __init__(
        self,
        endpoint_url: str = "http://localhost:8000",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        self.endpoint_url = endpoint_url.rstrip('/')
        self.access_key = access_key or os.environ.get("LAKEFS_ACCESS_KEY")
        self.secret_key = secret_key or os.environ.get("LAKEFS_SECRET_KEY")

        if not self.access_key or not self.secret_key:
            logger.warning("lakeFS credentials not provided")

        self.session = requests.Session()
        if self.access_key and self.secret_key:
            self.session.auth = (self.access_key, self.secret_key)

        logger.info(f"lakeFS client initialized: {endpoint_url}")

    def create_repository(
        self,
        name: str,
        storage_namespace: str,
        default_branch: str = "main",
    ) -> Dict[str, Any]:
        """
        Create a new repository.

        Args:
            name: Repository name
            storage_namespace: S3/Azure/GCS path (e.g., s3://my-bucket/path)
            default_branch: Default branch name

        Returns:
            Repository information
        """
        url = f"{self.endpoint_url}/api/v1/repositories"

        payload = {
            "name": name,
            "storage_namespace": storage_namespace,
            "default_branch": default_branch,
        }

        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()

            repo_info = response.json()
            logger.info(f"Repository created: {name}")

            return repo_info

        except requests.HTTPError as e:
            if e.response.status_code == 409:
                logger.warning(f"Repository {name} already exists")
                return self.get_repository(name)
            raise

    def get_repository(self, name: str) -> Dict[str, Any]:
        """Get repository information."""
        url = f"{self.endpoint_url}/api/v1/repositories/{name}"

        response = self.session.get(url)
        response.raise_for_status()

        return response.json()

    def list_repositories(self) -> List[Dict[str, Any]]:
        """List all repositories."""
        url = f"{self.endpoint_url}/api/v1/repositories"

        response = self.session.get(url)
        response.raise_for_status()

        result = response.json()
        return result.get('results', [])

    def create_branch(
        self,
        repository: str,
        name: str,
        source_branch: str = "main",
    ) -> LakeFSBranch:
        """
        Create a new branch.

        Args:
            repository: Repository name
            name: Branch name
            source_branch: Source branch to branch from

        Returns:
            Branch information
        """
        url = f"{self.endpoint_url}/api/v1/repositories/{repository}/branches"

        payload = {
            "name": name,
            "source": source_branch,
        }

        response = self.session.post(url, json=payload)
        response.raise_for_status()

        logger.info(f"Branch created: {repository}/{name}")

        # Get branch info
        return self.get_branch(repository, name)

    def get_branch(self, repository: str, name: str) -> LakeFSBranch:
        """Get branch information."""
        url = f"{self.endpoint_url}/api/v1/repositories/{repository}/branches/{name}"

        response = self.session.get(url)
        response.raise_for_status()

        data = response.json()

        return LakeFSBranch(
            name=name,
            commit_id=data.get('commit_id', ''),
            created_at=data.get('created_at', ''),
        )

    def list_branches(self, repository: str) -> List[LakeFSBranch]:
        """List all branches in a repository."""
        url = f"{self.endpoint_url}/api/v1/repositories/{repository}/branches"

        response = self.session.get(url)
        response.raise_for_status()

        result = response.json()
        branches = []

        for branch_data in result.get('results', []):
            branches.append(LakeFSBranch(
                name=branch_data['id'],
                commit_id=branch_data.get('commit_id', ''),
                created_at='',
            ))

        return branches

    def commit(
        self,
        repository: str,
        branch: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LakeFSCommit:
        """
        Commit changes on a branch.

        Args:
            repository: Repository name
            branch: Branch name
            message: Commit message
            metadata: Additional metadata

        Returns:
            Commit information
        """
        url = f"{self.endpoint_url}/api/v1/repositories/{repository}/branches/{branch}/commits"

        payload = {
            "message": message,
            "metadata": metadata or {},
        }

        response = self.session.post(url, json=payload)
        response.raise_for_status()

        data = response.json()

        commit = LakeFSCommit(
            id=data['id'],
            message=message,
            committer=data.get('committer', 'unknown'),
            creation_date=data.get('creation_date', ''),
            metadata=metadata or {},
            parents=data.get('parents', []),
        )

        logger.info(f"Commit created: {repository}/{branch} ({commit.id[:8]})")

        return commit

    def get_commit(
        self,
        repository: str,
        commit_id: str,
    ) -> LakeFSCommit:
        """Get commit information."""
        url = f"{self.endpoint_url}/api/v1/repositories/{repository}/commits/{commit_id}"

        response = self.session.get(url)
        response.raise_for_status()

        data = response.json()

        return LakeFSCommit(
            id=data['id'],
            message=data.get('message', ''),
            committer=data.get('committer', 'unknown'),
            creation_date=data.get('creation_date', ''),
            metadata=data.get('metadata', {}),
            parents=data.get('parents', []),
        )

    def merge(
        self,
        repository: str,
        source_branch: str,
        destination_branch: str,
        message: Optional[str] = None,
    ) -> LakeFSCommit:
        """
        Merge one branch into another.

        Args:
            repository: Repository name
            source_branch: Source branch
            destination_branch: Destination branch
            message: Merge commit message

        Returns:
            Merge commit
        """
        url = f"{self.endpoint_url}/api/v1/repositories/{repository}/refs/{destination_branch}/merge/{source_branch}"

        payload = {
            "message": message or f"Merge {source_branch} into {destination_branch}",
        }

        response = self.session.post(url, json=payload)
        response.raise_for_status()

        data = response.json()

        logger.info(f"Merged {source_branch} → {destination_branch}")

        return LakeFSCommit(
            id=data.get('reference', ''),
            message=payload['message'],
            committer='system',
            creation_date=datetime.now().isoformat(),
            metadata={},
            parents=[],
        )

    def upload_object(
        self,
        repository: str,
        branch: str,
        path: str,
        content: Union[bytes, str],
        content_type: str = "application/octet-stream",
    ) -> Dict[str, Any]:
        """
        Upload an object to lakeFS.

        Args:
            repository: Repository name
            branch: Branch name
            path: Object path
            content: Object content
            content_type: Content type

        Returns:
            Upload result
        """
        url = f"{self.endpoint_url}/api/v1/repositories/{repository}/branches/{branch}/objects"

        if isinstance(content, str):
            content = content.encode('utf-8')

        params = {"path": path}
        headers = {"Content-Type": content_type}

        response = self.session.post(url, params=params, data=content, headers=headers)
        response.raise_for_status()

        logger.info(f"Object uploaded: {repository}/{branch}/{path}")

        return response.json()

    def get_object(
        self,
        repository: str,
        ref: str,  # branch or commit
        path: str,
    ) -> bytes:
        """
        Get an object from lakeFS.

        Args:
            repository: Repository name
            ref: Branch or commit ID
            path: Object path

        Returns:
            Object content
        """
        url = f"{self.endpoint_url}/api/v1/repositories/{repository}/refs/{ref}/objects"

        params = {"path": path}

        response = self.session.get(url, params=params)
        response.raise_for_status()

        return response.content

    def list_objects(
        self,
        repository: str,
        ref: str,
        path: str = "",
    ) -> List[Dict[str, Any]]:
        """
        List objects in a path.

        Args:
            repository: Repository name
            ref: Branch or commit ID
            path: Path prefix

        Returns:
            List of objects
        """
        url = f"{self.endpoint_url}/api/v1/repositories/{repository}/refs/{ref}/objects/ls"

        params = {"prefix": path}

        response = self.session.get(url, params=params)
        response.raise_for_status()

        result = response.json()
        return result.get('results', [])

    def diff(
        self,
        repository: str,
        left_ref: str,
        right_ref: str,
    ) -> List[Dict[str, Any]]:
        """
        Get diff between two refs.

        Args:
            repository: Repository name
            left_ref: Left ref (branch/commit)
            right_ref: Right ref (branch/commit)

        Returns:
            List of changes
        """
        url = f"{self.endpoint_url}/api/v1/repositories/{repository}/refs/{left_ref}/diff/{right_ref}"

        response = self.session.get(url)
        response.raise_for_status()

        result = response.json()
        return result.get('results', [])

    def version_dataset(
        self,
        repository: str,
        dataset_path: str,
        branch: str = "main",
        commit_message: Optional[str] = None,
    ) -> LakeFSDatasetVersion:
        """
        Version a dataset (helper method).

        Args:
            repository: Repository name
            dataset_path: Local path to dataset
            branch: Branch to commit to
            commit_message: Commit message

        Returns:
            Dataset version metadata
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise ValueError(f"Dataset path not found: {dataset_path}")

        # Upload all files
        total_size = 0
        num_files = 0

        for file_path in dataset_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(dataset_path)
                lakefs_path = f"datasets/{dataset_path.name}/{relative_path}"

                with open(file_path, 'rb') as f:
                    content = f.read()

                self.upload_object(
                    repository=repository,
                    branch=branch,
                    path=lakefs_path,
                    content=content,
                )

                total_size += len(content)
                num_files += 1

        # Commit
        commit = self.commit(
            repository=repository,
            branch=branch,
            message=commit_message or f"Add dataset: {dataset_path.name}",
            metadata={
                'dataset_name': dataset_path.name,
                'total_size_bytes': total_size,
                'num_files': num_files,
            },
        )

        # Create version metadata
        version = LakeFSDatasetVersion(
            repository=repository,
            branch=branch,
            commit_id=commit.id,
            path=f"datasets/{dataset_path.name}",
            size_bytes=total_size,
            num_files=num_files,
            created_at=commit.creation_date,
            metadata=commit.metadata,
        )

        logger.info(f"Dataset versioned: {dataset_path.name} ({num_files} files, {total_size / 1e6:.1f} MB)")

        return version

    def checkout_dataset(
        self,
        repository: str,
        commit_id: str,
        dataset_name: str,
        output_path: str,
    ) -> str:
        """
        Checkout a dataset version.

        Args:
            repository: Repository name
            commit_id: Commit ID to checkout
            dataset_name: Dataset name
            output_path: Local output path

        Returns:
            Output path
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # List objects
        dataset_path = f"datasets/{dataset_name}"
        objects = self.list_objects(repository, commit_id, dataset_path)

        # Download all files
        for obj in objects:
            if obj['type'] != 'object':
                continue

            obj_path = obj['path']
            relative_path = obj_path.replace(f"{dataset_path}/", "")

            content = self.get_object(repository, commit_id, obj_path)

            file_path = output_path / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'wb') as f:
                f.write(content)

        logger.info(f"Dataset checked out: {dataset_name} @ {commit_id[:8]} → {output_path}")

        return str(output_path)


# Integration with DVC (alternative approach)
class DVCIntegration:
    """
    DVC (Data Version Control) integration.

    Alternative to lakeFS for dataset versioning.
    """

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)

    def init(self):
        """Initialize DVC repository."""
        import subprocess

        subprocess.run(['dvc', 'init'], cwd=self.repo_path, check=True)
        logger.info("DVC initialized")

    def add(self, file_path: str):
        """Add file to DVC."""
        import subprocess

        subprocess.run(['dvc', 'add', file_path], cwd=self.repo_path, check=True)
        logger.info(f"DVC tracked: {file_path}")

    def push(self, remote: Optional[str] = None):
        """Push to DVC remote."""
        import subprocess

        cmd = ['dvc', 'push']
        if remote:
            cmd.extend(['-r', remote])

        subprocess.run(cmd, cwd=self.repo_path, check=True)
        logger.info("DVC push complete")

    def pull(self, remote: Optional[str] = None):
        """Pull from DVC remote."""
        import subprocess

        cmd = ['dvc', 'pull']
        if remote:
            cmd.extend(['-r', remote])

        subprocess.run(cmd, cwd=self.repo_path, check=True)
        logger.info("DVC pull complete")


# Example usage
if __name__ == "__main__":
    # Initialize lakeFS client
    client = LakeFSClient(
        endpoint_url="http://localhost:8000",
        access_key="AKIAIOSFODNN7EXAMPLE",
        secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )

    # Create repository
    repo = client.create_repository(
        name="datasets",
        storage_namespace="s3://my-bucket/lakefs",
    )

    # Create branch
    branch = client.create_branch(
        repository="datasets",
        name="experiment-1",
        source_branch="main",
    )

    # Version a dataset
    version = client.version_dataset(
        repository="datasets",
        dataset_path="/path/to/dataset",
        branch="experiment-1",
        commit_message="Initial dataset version",
    )

    print(f"Dataset version: {version.commit_id}")

    # Merge to main
    merge_commit = client.merge(
        repository="datasets",
        source_branch="experiment-1",
        destination_branch="main",
    )

    print(f"Merged: {merge_commit.id}")
