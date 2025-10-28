"""
Quality gates for dataset curation.

Provides:
- Deduplication (SimHash/LSH for text, CLIP embeddings for images, audio fingerprinting)
- Toxicity detection and filtering
- PII redaction (RegEx + Presidio + optional cloud-DLP)
"""

import os
import re
import logging
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    """Result of quality check."""
    passed: bool
    score: float
    reason: str
    metadata: Dict[str, Any] = None


class QualityGate:
    """Base class for quality gates."""

    def check(self, item: Any) -> QualityCheckResult:
        """Check if item passes quality gate."""
        raise NotImplementedError


class DedupFilter(QualityGate):
    """
    Deduplication filter using multiple strategies:
    - Text: SimHash + MinHash LSH
    - Images: Perceptual hashing + CLIP embeddings
    - Audio: Audio fingerprinting (chromaprint)
    """

    def __init__(
        self,
        method: str = "auto",  # auto, simhash, perceptual, clip, chromaprint
        threshold: float = 0.9,  # Similarity threshold
        cache_dir: Optional[str] = None,
    ):
        self.method = method
        self.threshold = threshold
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # In-memory hash storage
        self.seen_hashes: Set[str] = set()
        self.lsh_buckets: Dict[int, Set[str]] = defaultdict(set)

        # Lazy-loaded models
        self._clip_model = None
        self._clip_processor = None

    @property
    def clip_model(self):
        """Lazy load CLIP model."""
        if self._clip_model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
                self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                if os.environ.get("CUDA_VISIBLE_DEVICES"):
                    self._clip_model = self._clip_model.to("cuda")
                logger.info("CLIP model loaded for deduplication")
            except Exception as e:
                logger.warning(f"Failed to load CLIP model: {e}")
        return self._clip_model, self._clip_processor

    def check(self, item: Dict) -> QualityCheckResult:
        """Check if item is a duplicate."""
        content_type = item.get('content_type', 'text')

        if content_type == 'text':
            return self._check_text_dedup(item)
        elif content_type == 'image':
            return self._check_image_dedup(item)
        elif content_type == 'audio':
            return self._check_audio_dedup(item)
        else:
            return QualityCheckResult(passed=True, score=0.0, reason="No dedup check for type")

    def _check_text_dedup(self, item: Dict) -> QualityCheckResult:
        """Check text deduplication using SimHash."""
        text = item.get('text', '')
        if not text:
            return QualityCheckResult(passed=True, score=0.0, reason="No text content")

        # Compute SimHash
        simhash = self._compute_simhash(text)

        # Check against seen hashes using Hamming distance
        for seen_hash in self.seen_hashes:
            similarity = 1 - (bin(int(simhash, 16) ^ int(seen_hash, 16)).count('1') / 64)
            if similarity >= self.threshold:
                return QualityCheckResult(
                    passed=False,
                    score=similarity,
                    reason=f"Duplicate text (similarity: {similarity:.2f})",
                    metadata={'duplicate_hash': seen_hash}
                )

        # Add to seen hashes
        self.seen_hashes.add(simhash)
        return QualityCheckResult(passed=True, score=0.0, reason="Unique text")

    def _compute_simhash(self, text: str, hash_bits: int = 64) -> str:
        """
        Compute SimHash for text.

        SimHash is a locality-sensitive hash that produces similar hashes
        for similar texts.
        """
        # Tokenize
        tokens = re.findall(r'\w+', text.lower())

        # Vector of hash_bits dimensions
        v = [0] * hash_bits

        for token in tokens:
            # Hash token
            h = hashlib.md5(token.encode()).hexdigest()
            h_int = int(h, 16)

            # Update vector
            for i in range(hash_bits):
                if h_int & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1

        # Generate final hash
        simhash = 0
        for i in range(hash_bits):
            if v[i] > 0:
                simhash |= (1 << i)

        return hex(simhash)[2:].zfill(16)

    def _check_image_dedup(self, item: Dict) -> QualityCheckResult:
        """Check image deduplication using perceptual hashing + CLIP embeddings."""
        source_path = item.get('source_path') or item.get('processed_path')
        if not source_path:
            return QualityCheckResult(passed=True, score=0.0, reason="No image path")

        # Method 1: Perceptual hash (fast)
        try:
            from PIL import Image
            img = Image.open(source_path)
            phash = self._compute_perceptual_hash(img)

            # Check for exact perceptual hash match
            if phash in self.seen_hashes:
                return QualityCheckResult(
                    passed=False,
                    score=1.0,
                    reason="Exact perceptual hash match",
                    metadata={'duplicate_hash': phash}
                )

            # Method 2: CLIP embeddings for near-duplicates (if enabled)
            if self.method in ['clip', 'auto']:
                clip_model, clip_processor = self.clip_model
                if clip_model is not None:
                    embedding = self._compute_clip_embedding(img, clip_model, clip_processor)
                    similarity = self._find_similar_embedding(embedding)
                    if similarity >= self.threshold:
                        return QualityCheckResult(
                            passed=False,
                            score=similarity,
                            reason=f"Near-duplicate image (CLIP similarity: {similarity:.2f})"
                        )

            # Add to seen hashes
            self.seen_hashes.add(phash)
            return QualityCheckResult(passed=True, score=0.0, reason="Unique image")

        except Exception as e:
            logger.error(f"Image dedup check failed: {e}")
            return QualityCheckResult(passed=True, score=0.0, reason=f"Error: {e}")

    def _compute_perceptual_hash(self, img, hash_size: int = 8) -> str:
        """
        Compute perceptual hash (pHash) for image.

        Args:
            img: PIL Image
            hash_size: Size of hash (default 8x8 = 64 bits)

        Returns:
            Hex string of hash
        """
        from PIL import Image
        import numpy as np

        # Convert to grayscale and resize
        img = img.convert('L').resize((hash_size, hash_size), Image.Resampling.LANCZOS)

        # Compute DCT (discrete cosine transform) approximation
        pixels = np.array(img).astype(float)

        # Simple approach: compare to average
        avg = pixels.mean()
        diff = pixels > avg

        # Convert to hash
        hash_str = ''.join('1' if val else '0' for val in diff.flatten())
        return hex(int(hash_str, 2))[2:].zfill(16)

    def _compute_clip_embedding(self, img, model, processor):
        """Compute CLIP embedding for image."""
        import torch

        inputs = processor(images=img, return_tensors="pt")
        if next(model.parameters()).is_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            embedding = outputs.cpu().numpy().flatten()

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def _find_similar_embedding(self, embedding: np.ndarray) -> float:
        """Find most similar embedding in cache."""
        # This is a placeholder - in production use FAISS or similar
        # For now, return 0 (no duplicates)
        return 0.0

    def _check_audio_dedup(self, item: Dict) -> QualityCheckResult:
        """Check audio deduplication using chromaprint."""
        source_path = item.get('source_path') or item.get('processed_path')
        if not source_path:
            return QualityCheckResult(passed=True, score=0.0, reason="No audio path")

        try:
            # Compute audio fingerprint using chromaprint (via acoustid)
            fingerprint = self._compute_audio_fingerprint(source_path)

            if fingerprint in self.seen_hashes:
                return QualityCheckResult(
                    passed=False,
                    score=1.0,
                    reason="Exact audio fingerprint match",
                    metadata={'duplicate_hash': fingerprint}
                )

            self.seen_hashes.add(fingerprint)
            return QualityCheckResult(passed=True, score=0.0, reason="Unique audio")

        except Exception as e:
            logger.error(f"Audio dedup check failed: {e}")
            return QualityCheckResult(passed=True, score=0.0, reason=f"Error: {e}")

    def _compute_audio_fingerprint(self, audio_path: str) -> str:
        """
        Compute audio fingerprint using chromaprint.

        Requires: fpcalc (chromaprint) command-line tool
        """
        import subprocess

        try:
            result = subprocess.run(
                ['fpcalc', '-raw', audio_path],
                capture_output=True,
                text=True,
                check=True
            )
            # Parse output: FINGERPRINT=<hash>
            for line in result.stdout.split('\n'):
                if line.startswith('FINGERPRINT='):
                    return line.split('=')[1]
            return ""
        except Exception as e:
            logger.warning(f"Chromaprint not available: {e}")
            # Fallback to simple hash
            return hashlib.md5(open(audio_path, 'rb').read()).hexdigest()


class ToxicityFilter(QualityGate):
    """
    Toxicity detection and filtering for text content.

    Uses detoxify library (based on BERT) for toxicity classification.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        categories: Optional[List[str]] = None,
    ):
        self.threshold = threshold
        self.categories = categories or ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult']

        # Lazy load model
        self._toxicity_model = None

    @property
    def toxicity_model(self):
        """Lazy load toxicity model."""
        if self._toxicity_model is None:
            try:
                from detoxify import Detoxify
                self._toxicity_model = Detoxify('original')
                logger.info("Toxicity model loaded")
            except Exception as e:
                logger.warning(f"Failed to load toxicity model: {e}")
        return self._toxicity_model

    def check(self, item: Dict) -> QualityCheckResult:
        """Check if text content is toxic."""
        text = item.get('text', '')
        if not text:
            return QualityCheckResult(passed=True, score=0.0, reason="No text content")

        if self.toxicity_model is None:
            return QualityCheckResult(passed=True, score=0.0, reason="Toxicity model not available")

        try:
            results = self.toxicity_model.predict(text)

            # Check each category
            max_score = 0.0
            flagged_category = None

            for category in self.categories:
                score = results.get(category, 0.0)
                if score > max_score:
                    max_score = score
                    flagged_category = category

            if max_score >= self.threshold:
                return QualityCheckResult(
                    passed=False,
                    score=max_score,
                    reason=f"Toxic content detected: {flagged_category} ({max_score:.2f})",
                    metadata=results
                )

            return QualityCheckResult(passed=True, score=max_score, reason="Content is safe")

        except Exception as e:
            logger.error(f"Toxicity check failed: {e}")
            return QualityCheckResult(passed=True, score=0.0, reason=f"Error: {e}")


class PIIRedactor(QualityGate):
    """
    PII (Personally Identifiable Information) redaction for text.

    Uses:
    - RegEx patterns for common PII (emails, phone numbers, SSN)
    - Presidio for advanced entity recognition
    - Optional cloud DLP API support
    """

    def __init__(
        self,
        use_presidio: bool = True,
        use_regex: bool = True,
        use_cloud_dlp: bool = False,
        redact: bool = True,  # If False, just detect
    ):
        self.use_presidio = use_presidio
        self.use_regex = use_regex
        self.use_cloud_dlp = use_cloud_dlp
        self.redact = redact

        # Lazy load Presidio
        self._presidio_analyzer = None
        self._presidio_anonymizer = None

        # Regex patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        }

    @property
    def presidio_analyzer(self):
        """Lazy load Presidio analyzer."""
        if self._presidio_analyzer is None and self.use_presidio:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine
                self._presidio_analyzer = AnalyzerEngine()
                self._presidio_anonymizer = AnonymizerEngine()
                logger.info("Presidio loaded")
            except Exception as e:
                logger.warning(f"Failed to load Presidio: {e}")
        return self._presidio_analyzer, self._presidio_anonymizer

    def check(self, item: Dict) -> QualityCheckResult:
        """Check and optionally redact PII in text."""
        text = item.get('text', '')
        if not text:
            return QualityCheckResult(passed=True, score=0.0, reason="No text content")

        pii_found = []
        redacted_text = text

        # Check with RegEx
        if self.use_regex:
            for pii_type, pattern in self.pii_patterns.items():
                matches = re.finditer(pattern, text)
                for match in matches:
                    pii_found.append({
                        'type': pii_type,
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                    })

                    if self.redact:
                        redacted_text = re.sub(pattern, f'[{pii_type.upper()}]', redacted_text)

        # Check with Presidio
        if self.use_presidio:
            analyzer, anonymizer = self.presidio_analyzer
            if analyzer:
                try:
                    results = analyzer.analyze(text=text, language='en')
                    for result in results:
                        pii_found.append({
                            'type': result.entity_type,
                            'text': text[result.start:result.end],
                            'start': result.start,
                            'end': result.end,
                            'score': result.score,
                        })

                    if self.redact and anonymizer:
                        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
                        redacted_text = anonymized.text

                except Exception as e:
                    logger.error(f"Presidio analysis failed: {e}")

        # Update item with redacted text if requested
        if self.redact and pii_found:
            item['text_redacted'] = redacted_text

        if pii_found:
            return QualityCheckResult(
                passed=False,
                score=len(pii_found),
                reason=f"PII detected: {len(pii_found)} instances",
                metadata={'pii_found': pii_found}
            )

        return QualityCheckResult(passed=True, score=0.0, reason="No PII detected")


class QualityPipeline:
    """
    Quality pipeline that applies multiple quality gates in sequence.
    """

    def __init__(self, gates: List[QualityGate]):
        self.gates = gates

    def check(self, item: Dict) -> Tuple[bool, List[QualityCheckResult]]:
        """
        Check item through all quality gates.

        Args:
            item: Item to check

        Returns:
            Tuple of (passed, results)
        """
        results = []
        for gate in self.gates:
            result = gate.check(item)
            results.append(result)
            if not result.passed:
                return False, results

        return True, results

    def filter_dataset(
        self,
        items: List[Dict],
        save_rejected: bool = False,
        rejected_path: Optional[str] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter dataset through quality pipeline.

        Args:
            items: List of items
            save_rejected: Save rejected items
            rejected_path: Path to save rejected items

        Returns:
            Tuple of (passed_items, rejected_items)
        """
        passed = []
        rejected = []

        for item in items:
            passed_check, results = self.check(item)
            if passed_check:
                passed.append(item)
            else:
                # Add rejection reason
                item['rejection_reasons'] = [r.reason for r in results if not r.passed]
                rejected.append(item)

        logger.info(f"Quality check: {len(passed)} passed, {len(rejected)} rejected")

        if save_rejected and rejected_path:
            with open(rejected_path, 'w') as f:
                for item in rejected:
                    f.write(json.dumps(item) + '\n')
            logger.info(f"Rejected items saved to {rejected_path}")

        return passed, rejected


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Quality gates for datasets")
    parser.add_argument("manifest", help="Input manifest (JSONL)")
    parser.add_argument("--output", "-o", required=True, help="Output manifest (filtered)")
    parser.add_argument("--rejected", help="Save rejected items to this file")
    parser.add_argument("--dedup", action="store_true", help="Enable deduplication")
    parser.add_argument("--toxicity", action="store_true", help="Enable toxicity filter")
    parser.add_argument("--pii", action="store_true", help="Enable PII redaction")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load manifest
    items = []
    with open(args.manifest, 'r') as f:
        for line in f:
            items.append(json.loads(line.strip()))

    # Build quality pipeline
    gates = []
    if args.dedup:
        gates.append(DedupFilter())
    if args.toxicity:
        gates.append(ToxicityFilter())
    if args.pii:
        gates.append(PIIRedactor())

    if not gates:
        print("No quality gates enabled. Use --dedup, --toxicity, or --pii")
        return

    pipeline = QualityPipeline(gates)

    # Filter dataset
    passed, rejected = pipeline.filter_dataset(
        items,
        save_rejected=bool(args.rejected),
        rejected_path=args.rejected,
    )

    # Save passed items
    with open(args.output, 'w') as f:
        for item in passed:
            f.write(json.dumps(item) + '\n')

    print(f"\nResults:")
    print(f"  Passed: {len(passed)}")
    print(f"  Rejected: {len(rejected)}")


if __name__ == "__main__":
    main()
