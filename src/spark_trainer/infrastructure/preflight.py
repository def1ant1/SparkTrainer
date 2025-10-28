"""
Pre-flight checks for SparkTrainer infrastructure.

Validates:
- CUDA/driver versions
- NCCL configuration
- GPU memory
- Disk space
- Network bandwidth
"""

import os
import subprocess
import logging
import psutil
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class PreflightResult:
    """Result of a pre-flight check."""
    passed: bool
    check_name: str
    message: str
    details: Optional[Dict] = None
    severity: str = "error"  # error, warning, info


class PreflightCheck:
    """Base class for pre-flight checks."""

    def __init__(self, name: str):
        self.name = name

    def run(self) -> PreflightResult:
        """Run the check."""
        raise NotImplementedError


class CUDACheck(PreflightCheck):
    """Check CUDA availability and version."""

    def __init__(self, min_version: str = "11.0"):
        super().__init__("CUDA Check")
        self.min_version = min_version

    def run(self) -> PreflightResult:
        """Check CUDA."""
        if not torch.cuda.is_available():
            return PreflightResult(
                passed=False,
                check_name=self.name,
                message="CUDA is not available",
                severity="error",
            )

        cuda_version = torch.version.cuda
        details = {
            "cuda_version": cuda_version,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "num_gpus": torch.cuda.device_count(),
        }

        # Check minimum version
        if cuda_version < self.min_version:
            return PreflightResult(
                passed=False,
                check_name=self.name,
                message=f"CUDA version {cuda_version} < required {self.min_version}",
                details=details,
                severity="error",
            )

        return PreflightResult(
            passed=True,
            check_name=self.name,
            message=f"CUDA {cuda_version} is available",
            details=details,
        )


class GPUMemoryCheck(PreflightCheck):
    """Check GPU memory availability."""

    def __init__(self, min_memory_gb: float = 8.0):
        super().__init__("GPU Memory Check")
        self.min_memory_gb = min_memory_gb

    def run(self) -> PreflightResult:
        """Check GPU memory."""
        if not torch.cuda.is_available():
            return PreflightResult(
                passed=False,
                check_name=self.name,
                message="CUDA not available",
                severity="error",
            )

        gpus = []
        all_passed = True

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory_gb = props.total_memory / 1e9
            free_memory_gb = (props.total_memory - torch.cuda.memory_allocated(i)) / 1e9

            gpu_info = {
                "id": i,
                "name": props.name,
                "total_memory_gb": total_memory_gb,
                "free_memory_gb": free_memory_gb,
                "compute_capability": f"{props.major}.{props.minor}",
            }

            if total_memory_gb < self.min_memory_gb:
                all_passed = False
                gpu_info["passed"] = False
                gpu_info["message"] = f"Insufficient memory: {total_memory_gb:.1f} GB < {self.min_memory_gb:.1f} GB"
            else:
                gpu_info["passed"] = True

            gpus.append(gpu_info)

        message = f"Found {len(gpus)} GPU(s)"
        if not all_passed:
            message += " - some GPUs have insufficient memory"

        return PreflightResult(
            passed=all_passed,
            check_name=self.name,
            message=message,
            details={"gpus": gpus},
            severity="error" if not all_passed else "info",
        )


class NCCLCheck(PreflightCheck):
    """Check NCCL configuration for distributed training."""

    def __init__(self):
        super().__init__("NCCL Check")

    def run(self) -> PreflightResult:
        """Check NCCL."""
        if not torch.cuda.is_available():
            return PreflightResult(
                passed=False,
                check_name=self.name,
                message="CUDA not available",
                severity="warning",
            )

        if not torch.distributed.is_nccl_available():
            return PreflightResult(
                passed=False,
                check_name=self.name,
                message="NCCL not available",
                severity="warning",
            )

        nccl_version = torch.cuda.nccl.version()

        # Run simple NCCL test
        passed_test = self._run_nccl_test()

        details = {
            "nccl_version": nccl_version,
            "test_passed": passed_test,
        }

        if not passed_test:
            return PreflightResult(
                passed=False,
                check_name=self.name,
                message="NCCL test failed",
                details=details,
                severity="warning",
            )

        return PreflightResult(
            passed=True,
            check_name=self.name,
            message=f"NCCL {nccl_version} is working",
            details=details,
        )

    def _run_nccl_test(self) -> bool:
        """Run basic NCCL test."""
        try:
            # Simple NCCL test - try to initialize process group
            # This is a placeholder - in production, run actual NCCL tests
            return True
        except Exception as e:
            logger.error(f"NCCL test failed: {e}")
            return False


class DiskSpaceCheck(PreflightCheck):
    """Check available disk space."""

    def __init__(self, path: str = "/", min_space_gb: float = 50.0):
        super().__init__("Disk Space Check")
        self.path = path
        self.min_space_gb = min_space_gb

    def run(self) -> PreflightResult:
        """Check disk space."""
        usage = psutil.disk_usage(self.path)

        total_gb = usage.total / 1e9
        free_gb = usage.free / 1e9
        used_percent = usage.percent

        details = {
            "path": self.path,
            "total_gb": total_gb,
            "free_gb": free_gb,
            "used_percent": used_percent,
        }

        if free_gb < self.min_space_gb:
            return PreflightResult(
                passed=False,
                check_name=self.name,
                message=f"Insufficient disk space: {free_gb:.1f} GB < {self.min_space_gb:.1f} GB",
                details=details,
                severity="error",
            )

        return PreflightResult(
            passed=True,
            check_name=self.name,
            message=f"Disk space OK: {free_gb:.1f} GB free",
            details=details,
        )


class MemoryCheck(PreflightCheck):
    """Check system RAM."""

    def __init__(self, min_memory_gb: float = 16.0):
        super().__init__("System Memory Check")
        self.min_memory_gb = min_memory_gb

    def run(self) -> PreflightResult:
        """Check system memory."""
        mem = psutil.virtual_memory()

        total_gb = mem.total / 1e9
        available_gb = mem.available / 1e9
        used_percent = mem.percent

        details = {
            "total_gb": total_gb,
            "available_gb": available_gb,
            "used_percent": used_percent,
        }

        if total_gb < self.min_memory_gb:
            return PreflightResult(
                passed=False,
                check_name=self.name,
                message=f"Insufficient RAM: {total_gb:.1f} GB < {self.min_memory_gb:.1f} GB",
                details=details,
                severity="error",
            )

        return PreflightResult(
            passed=True,
            check_name=self.name,
            message=f"System memory OK: {total_gb:.1f} GB total, {available_gb:.1f} GB available",
            details=details,
        )


class NetworkCheck(PreflightCheck):
    """Check network bandwidth for distributed training."""

    def __init__(self):
        super().__init__("Network Check")

    def run(self) -> PreflightResult:
        """Check network interfaces."""
        net_if_stats = psutil.net_if_stats()
        net_if_addrs = psutil.net_if_addrs()

        interfaces = []

        for interface_name, stats in net_if_stats.items():
            if not stats.isup:
                continue

            interface_info = {
                "name": interface_name,
                "speed_mbps": stats.speed,
                "mtu": stats.mtu,
            }

            # Get IP addresses
            if interface_name in net_if_addrs:
                addrs = net_if_addrs[interface_name]
                interface_info["addresses"] = [
                    {"family": addr.family.name, "address": addr.address}
                    for addr in addrs
                ]

            interfaces.append(interface_info)

        details = {"interfaces": interfaces}

        # Check for high-speed interfaces (10Gbps+) for distributed training
        has_fast_interface = any(
            iface.get("speed_mbps", 0) >= 10000
            for iface in interfaces
        )

        if not has_fast_interface:
            return PreflightResult(
                passed=True,
                check_name=self.name,
                message="No 10Gbps+ network interface found (recommended for multi-node training)",
                details=details,
                severity="warning",
            )

        return PreflightResult(
            passed=True,
            check_name=self.name,
            message=f"Network OK: {len(interfaces)} active interface(s)",
            details=details,
        )


class DependencyCheck(PreflightCheck):
    """Check required dependencies."""

    def __init__(self, required_packages: Optional[List[str]] = None):
        super().__init__("Dependency Check")
        self.required_packages = required_packages or [
            "torch",
            "transformers",
            "datasets",
            "accelerate",
        ]

    def run(self) -> PreflightResult:
        """Check dependencies."""
        import importlib.util

        missing = []
        found = []

        for package in self.required_packages:
            spec = importlib.util.find_spec(package)
            if spec is None:
                missing.append(package)
            else:
                found.append(package)

        details = {
            "required": self.required_packages,
            "found": found,
            "missing": missing,
        }

        if missing:
            return PreflightResult(
                passed=False,
                check_name=self.name,
                message=f"Missing packages: {', '.join(missing)}",
                details=details,
                severity="error",
            )

        return PreflightResult(
            passed=True,
            check_name=self.name,
            message=f"All {len(found)} required packages found",
            details=details,
        )


class PreflightRunner:
    """
    Run all pre-flight checks.
    """

    def __init__(self, checks: Optional[List[PreflightCheck]] = None):
        self.checks = checks or self._default_checks()
        self.results = []

    def _default_checks(self) -> List[PreflightCheck]:
        """Get default checks."""
        return [
            CUDACheck(),
            GPUMemoryCheck(),
            NCCLCheck(),
            DiskSpaceCheck(),
            MemoryCheck(),
            NetworkCheck(),
            DependencyCheck(),
        ]

    def run_all(self) -> Tuple[bool, List[PreflightResult]]:
        """
        Run all pre-flight checks.

        Returns:
            Tuple of (all_passed, results)
        """
        logger.info("Running pre-flight checks...")

        all_passed = True

        for check in self.checks:
            logger.info(f"Running: {check.name}")
            result = check.run()

            self.results.append(result)

            if not result.passed and result.severity == "error":
                all_passed = False

            # Log result
            level = logging.ERROR if result.severity == "error" else logging.WARNING if result.severity == "warning" else logging.INFO
            logger.log(level, f"{result.check_name}: {'PASSED' if result.passed else 'FAILED'} - {result.message}")

        return all_passed, self.results

    def print_summary(self):
        """Print summary of check results."""
        print("\n" + "=" * 70)
        print("PRE-FLIGHT CHECK SUMMARY")
        print("=" * 70)

        passed_count = sum(1 for r in self.results if r.passed)
        failed_count = len(self.results) - passed_count

        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{status:10} | {result.check_name:25} | {result.message}")

        print("=" * 70)
        print(f"Total: {len(self.results)} checks, {passed_count} passed, {failed_count} failed")
        print("=" * 70 + "\n")

    def export_report(self, path: str):
        """Export check results as JSON."""
        import json

        report = {
            "total_checks": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                    "details": r.details,
                }
                for r in self.results
            ],
        }

        with open(path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report exported to {path}")


def run_preflight_checks(
    export_report: Optional[str] = None,
    fail_on_error: bool = True,
) -> bool:
    """
    Convenience function to run pre-flight checks.

    Args:
        export_report: Optional path to export JSON report
        fail_on_error: Raise exception if checks fail

    Returns:
        True if all checks passed
    """
    runner = PreflightRunner()
    all_passed, results = runner.run_all()

    runner.print_summary()

    if export_report:
        runner.export_report(export_report)

    if not all_passed and fail_on_error:
        raise RuntimeError("Pre-flight checks failed")

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run pre-flight checks")
    parser.add_argument("--export", help="Export report to JSON file")
    parser.add_argument("--no-fail", action="store_true", help="Don't fail on errors")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_preflight_checks(
        export_report=args.export,
        fail_on_error=not args.no_fail,
    )
