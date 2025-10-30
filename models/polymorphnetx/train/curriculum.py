
from typing import Dict, Any
def filter_fn(obj: Dict[str, Any], stage: str) -> bool:
    text: str = obj.get("prompt", "")
    if stage == "pretrain": return True
    if stage == "dag": return any(t in text for t in ("[AGENT]", "[MEM]", "[PLUGIN]", "[VERIFY]"))
    if stage == "policy": return "[POLICY]" in text
    return True
