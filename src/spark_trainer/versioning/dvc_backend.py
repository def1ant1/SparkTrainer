"""
DVC (Data Version Control) backend for dataset versioning.

Supports:
- S3/MinIO remote storage
- Dataset versioning and lineage
- Checksum-based tracking
- Manifest generation
"""

import os
import json
import logging
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class DVCBackend:
    """
    DVC backend for dataset versioning.

    Provides:
    - Dataset versioning with DVC
    - S3/MinIO remote storage
    - Lineage tracking
    - Checksum validation
    """

    def __init__(
        self,
        repo_path: str,
        remote_url: Optional[str] = None,
        remote_name: str = "storage",
    ):
        """
        Initialize DVC backend.

        Args:
            repo_path: Path to DVC repository
            remote_url: S3/MinIO remote URL (e.g., s3://bucket/path)
            remote_name: Name for the remote
        """
        self.repo_path = Path(repo_path)
        self.remote_url = remote_url
        self.remote_name = remote_name

        # Initialize DVC repo if not exists
        if not (self.repo_path / ".dvc").exists():
            self._init_repo()

        # Configure remote if provided
        if remote_url:
            self._configure_remote(remote_url, remote_name)

    def _init_repo(self):
        """Initialize DVC repository."""
        logger.info(f"Initializing DVC repo at {self.repo_path}")

        # Initialize git repo first (DVC requires git)
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            subprocess.run(
                ["git", "init"],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            logger.info("Git repository initialized")

        # Initialize DVC
        subprocess.run(
            ["dvc", "init"],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )
        logger.info("DVC repository initialized")

    def _configure_remote(self, remote_url: str, remote_name: str):
        """Configure DVC remote storage."""
        logger.info(f"Configuring DVC remote '{remote_name}': {remote_url}")

        # Add remote
        subprocess.run(
            ["dvc", "remote", "add", "-f", remote_name, remote_url],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

        # Set as default
        subprocess.run(
            ["dvc", "remote", "default", remote_name],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

        # Configure S3/MinIO credentials if using S3
        if remote_url.startswith("s3://"):
            # Get credentials from environment
            access_key = os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            endpoint_url = os.getenv("AWS_ENDPOINT_URL")  # For MinIO

            if access_key:
                subprocess.run(
                    ["dvc", "remote", "modify", remote_name, "access_key_id", access_key],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True
                )

            if secret_key:
                subprocess.run(
                    ["dvc", "remote", "modify", remote_name, "secret_access_key", secret_key],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True
                )

            if endpoint_url:
                subprocess.run(
                    ["dvc", "remote", "modify", remote_name, "endpointurl", endpoint_url],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True
                )

        logger.info("DVC remote configured")

    def add_dataset(
        self,
        dataset_path: str,
        dataset_name: Optional[str] = None,
        commit_message: Optional[str] = None,
    ) -> str:
        """
        Add dataset to DVC tracking.

        Args:
            dataset_path: Path to dataset (file or directory)
            dataset_name: Optional dataset name
            commit_message: Git commit message

        Returns:
            DVC file path (.dvc)
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"Dataset not found: {dataset_path}")

        dataset_name = dataset_name or dataset_path.name

        # Make path relative to repo
        if dataset_path.is_absolute():
            try:
                rel_path = dataset_path.relative_to(self.repo_path)
            except ValueError:
                raise ValueError(f"Dataset path must be within repo: {dataset_path}")
        else:
            rel_path = dataset_path

        logger.info(f"Adding dataset to DVC: {rel_path}")

        # Add to DVC
        result = subprocess.run(
            ["dvc", "add", str(rel_path)],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"DVC add failed: {result.stderr}")

        dvc_file = self.repo_path / f"{rel_path}.dvc"

        # Commit to git
        commit_msg = commit_message or f"Add dataset: {dataset_name}"
        subprocess.run(
            ["git", "add", str(dvc_file), ".gitignore"],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

        logger.info(f"Dataset added and committed: {dvc_file}")

        return str(dvc_file)

    def push(self, remote_name: Optional[str] = None):
        """Push datasets to remote storage."""
        remote = remote_name or self.remote_name

        logger.info(f"Pushing datasets to remote '{remote}'")

        result = subprocess.run(
            ["dvc", "push", "-r", remote],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"DVC push failed: {result.stderr}")

        logger.info("Datasets pushed successfully")

    def pull(self, remote_name: Optional[str] = None):
        """Pull datasets from remote storage."""
        remote = remote_name or self.remote_name

        logger.info(f"Pulling datasets from remote '{remote}'")

        result = subprocess.run(
            ["dvc", "pull", "-r", remote],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"DVC pull failed: {result.stderr}")

        logger.info("Datasets pulled successfully")

    def checkout(self, version: str, dataset_path: Optional[str] = None):
        """
        Checkout specific version of dataset.

        Args:
            version: Git commit hash or tag
            dataset_path: Optional specific dataset to checkout
        """
        logger.info(f"Checking out version: {version}")

        # Git checkout
        subprocess.run(
            ["git", "checkout", version],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

        # DVC checkout
        if dataset_path:
            subprocess.run(
                ["dvc", "checkout", dataset_path],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
        else:
            subprocess.run(
                ["dvc", "checkout"],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )

        logger.info("Checkout completed")

    def get_lineage(self, dataset_path: str) -> List[Dict]:
        """
        Get lineage graph for dataset.

        Args:
            dataset_path: Path to dataset

        Returns:
            List of version history entries
        """
        dataset_path = Path(dataset_path)
        dvc_file = f"{dataset_path}.dvc"

        # Get git log for .dvc file
        result = subprocess.run(
            ["git", "log", "--format=%H|%at|%s", "--", dvc_file],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )

        lineage = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            commit_hash, timestamp, message = line.split('|', 2)
            lineage.append({
                'commit': commit_hash,
                'timestamp': int(timestamp),
                'datetime': datetime.fromtimestamp(int(timestamp)).isoformat(),
                'message': message,
            })

        return lineage

    def get_checksum(self, dataset_path: str) -> Optional[str]:
        """
        Get MD5 checksum for dataset from DVC file.

        Args:
            dataset_path: Path to dataset

        Returns:
            MD5 checksum or None
        """
        dvc_file = self.repo_path / f"{dataset_path}.dvc"
        if not dvc_file.exists():
            return None

        with open(dvc_file, 'r') as f:
            dvc_data = json.load(f) if dvc_file.suffix == '.json' else {}

        # DVC stores MD5 in 'md5' or 'outs' field
        if 'md5' in dvc_data:
            return dvc_data['md5']
        elif 'outs' in dvc_data and dvc_data['outs']:
            return dvc_data['outs'][0].get('md5')

        return None

    def create_version(
        self,
        dataset_path: str,
        version_tag: str,
        message: Optional[str] = None,
    ) -> str:
        """
        Create a version tag for dataset.

        Args:
            dataset_path: Path to dataset
            version_tag: Version tag (e.g., v1.0.0)
            message: Tag message

        Returns:
            Commit hash
        """
        logger.info(f"Creating version tag: {version_tag}")

        # Create git tag
        tag_msg = message or f"Version {version_tag}"
        subprocess.run(
            ["git", "tag", "-a", version_tag, "-m", tag_msg],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )

        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )

        commit_hash = result.stdout.strip()
        logger.info(f"Version tag created: {version_tag} @ {commit_hash}")

        return commit_hash

    def list_versions(self) -> List[Dict]:
        """
        List all version tags.

        Returns:
            List of version info
        """
        result = subprocess.run(
            ["git", "tag", "-l", "-n1"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )

        versions = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            parts = line.split(maxsplit=1)
            tag = parts[0]
            message = parts[1] if len(parts) > 1 else ""

            # Get commit info for tag
            commit_result = subprocess.run(
                ["git", "rev-list", "-n", "1", tag],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            commit = commit_result.stdout.strip()

            versions.append({
                'tag': tag,
                'commit': commit,
                'message': message,
            })

        return versions

    def diff_versions(
        self,
        version1: str,
        version2: str,
        dataset_path: str,
    ) -> Dict:
        """
        Compare two versions of a dataset.

        Args:
            version1: First version (commit/tag)
            version2: Second version (commit/tag)
            dataset_path: Path to dataset

        Returns:
            Diff information
        """
        dvc_file = f"{dataset_path}.dvc"

        # Get checksums for both versions
        def get_checksum_at_version(version):
            result = subprocess.run(
                ["git", "show", f"{version}:{dvc_file}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                dvc_data = json.loads(result.stdout)
                if 'outs' in dvc_data and dvc_data['outs']:
                    return dvc_data['outs'][0].get('md5')
            return None

        checksum1 = get_checksum_at_version(version1)
        checksum2 = get_checksum_at_version(version2)

        return {
            'version1': version1,
            'version2': version2,
            'checksum1': checksum1,
            'checksum2': checksum2,
            'changed': checksum1 != checksum2,
        }


class LakeFSBackend:
    """
    LakeFS backend for dataset versioning (alternative to DVC).

    LakeFS provides Git-like operations for data lakes.
    """

    def __init__(
        self,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        repository: str,
    ):
        """
        Initialize LakeFS backend.

        Args:
            endpoint_url: LakeFS API endpoint
            access_key_id: Access key
            secret_access_key: Secret key
            repository: Repository name
        """
        self.endpoint_url = endpoint_url.rstrip('/')
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.repository = repository

        # Will use lakefs Python client or REST API
        self._client = None

    @property
    def client(self):
        """Lazy load LakeFS client."""
        if self._client is None:
            try:
                import lakefs_client
                from lakefs_client.client import LakeFSClient

                config = lakefs_client.Configuration(
                    host=self.endpoint_url,
                    username=self.access_key_id,
                    password=self.secret_access_key,
                )

                self._client = LakeFSClient(config)
                logger.info("LakeFS client initialized")
            except ImportError:
                logger.warning("lakefs-client not installed. Install with: pip install lakefs-client")
            except Exception as e:
                logger.error(f"Failed to initialize LakeFS client: {e}")

        return self._client

    def create_branch(self, branch_name: str, source_branch: str = "main") -> Dict:
        """
        Create a new branch.

        Args:
            branch_name: Name of new branch
            source_branch: Source branch to branch from

        Returns:
            Branch info
        """
        if not self.client:
            raise RuntimeError("LakeFS client not available")

        logger.info(f"Creating branch: {branch_name} from {source_branch}")

        branch = self.client.branches.create_branch(
            repository=self.repository,
            branch_creation={
                'name': branch_name,
                'source': source_branch,
            }
        )

        return {
            'name': branch.id,
            'commit_id': branch.commit_id,
        }

    def commit(self, branch: str, message: str, metadata: Optional[Dict] = None) -> str:
        """
        Commit changes to branch.

        Args:
            branch: Branch name
            message: Commit message
            metadata: Optional metadata

        Returns:
            Commit ID
        """
        if not self.client:
            raise RuntimeError("LakeFS client not available")

        logger.info(f"Committing to branch: {branch}")

        commit = self.client.commits.commit(
            repository=self.repository,
            branch=branch,
            commit_creation={
                'message': message,
                'metadata': metadata or {},
            }
        )

        logger.info(f"Committed: {commit.id}")
        return commit.id

    def merge(self, source_branch: str, destination_branch: str) -> str:
        """
        Merge source branch into destination.

        Args:
            source_branch: Source branch
            destination_branch: Destination branch

        Returns:
            Merge commit ID
        """
        if not self.client:
            raise RuntimeError("LakeFS client not available")

        logger.info(f"Merging {source_branch} into {destination_branch}")

        result = self.client.refs.merge_into_branch(
            repository=self.repository,
            source_ref=source_branch,
            destination_branch=destination_branch,
        )

        return result.reference

    # Additional LakeFS operations...


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Dataset versioning with DVC")
    parser.add_argument("--repo", required=True, help="Repository path")
    parser.add_argument("--remote", help="Remote URL (s3://bucket/path)")
    parser.add_argument("--action", choices=['add', 'push', 'pull', 'versions', 'lineage'], required=True)
    parser.add_argument("--dataset", help="Dataset path")
    parser.add_argument("--tag", help="Version tag")
    parser.add_argument("--message", "-m", help="Commit/tag message")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    dvc = DVCBackend(
        repo_path=args.repo,
        remote_url=args.remote,
    )

    if args.action == 'add':
        if not args.dataset:
            print("--dataset required for add action")
            return
        dvc_file = dvc.add_dataset(args.dataset, commit_message=args.message)
        print(f"Dataset added: {dvc_file}")

    elif args.action == 'push':
        dvc.push()
        print("Datasets pushed")

    elif args.action == 'pull':
        dvc.pull()
        print("Datasets pulled")

    elif args.action == 'versions':
        versions = dvc.list_versions()
        print(f"\nVersions ({len(versions)}):")
        for v in versions:
            print(f"  {v['tag']} @ {v['commit'][:8]} - {v['message']}")

    elif args.action == 'lineage':
        if not args.dataset:
            print("--dataset required for lineage action")
            return
        lineage = dvc.get_lineage(args.dataset)
        print(f"\nLineage for {args.dataset} ({len(lineage)} commits):")
        for entry in lineage:
            print(f"  {entry['commit'][:8]} - {entry['datetime']} - {entry['message']}")


if __name__ == "__main__":
    main()
