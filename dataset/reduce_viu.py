import re
from typing import Dict, List, Optional, Tuple

# --------- Helpers ---------
CANON_KEYS = [
    "Product name",
    "Category",
    "Type/form",
    "Brand",
    "Packaging (primary color, secondary color, container type, closure type)",
    "Size/volume",
    "On-pack claims",
]

# Keywords that usually help ranking (keep if present)
RANK_CLAIM_KEYWORDS = [
    "spf", "broad spectrum", "uva", "uvb", "water resistant",
    "2-in-1", "2 in 1", "3-in-1", "3 in 1",
    "anti-dandruff", "antidandruff", "dandruff",
    "leave-in", "leave in",
    "keratin", "argan", "moroccan", "coconut", "aloe",
    "sensitive skin", "hypoallergenic", "fragrance-free", "unscented",
    "men+care", "men", "kids", "baby",
    "color care", "color protect", "damage repair", "frizz control",
    "deep conditioner", "mask", "serum", "mousse", "dry shampoo",
]

# Noisy marketing terms (often hallucinated / low ranking value)
NOISY_CLAIM_PATTERNS = [
    r"\bdetox\b",
    r"\bhair growth\b",
    r"\banti[- ]?aging\b",
    r"\b100% natural\b",
    r"\bnon[- ]?toxic\b",
    r"\beco[- ]?friendly\b",
    r"\bbiodegradable\b",
    r"\bcompostable\b",
    r"\bcruelty[- ]?free\b",
    r"\bvegan\b",
    r"\bparaben[- ]?free\b",   # keep sometimes, but often hallucinated
    r"\bdermatologist tested\b",  # often hallucinated
]

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if s.lower() == "unknown":
        return ""
    return _norm_space(s)

def _parse_viu_bullets(viu_text: str) -> Dict[str, object]:
    """
    Parse a bullet-style VIU into dict.
    Accepts:
      - key: value
      - key:
          - claim
    Returns dict with keys in CANON_KEYS if detected.
    """
    out: Dict[str, object] = {}
    if not isinstance(viu_text, str) or not viu_text.strip():
        return out

    lines = [ln.rstrip() for ln in viu_text.splitlines() if ln.strip()]
    key_re = re.compile(r"^\s*[-•*]?\s*([^:]{2,120})\s*:\s*(.*)$")
    bullet_re = re.compile(r"^\s*[-•*]\s+(.*)$")

    cur_key: Optional[str] = None

    for ln in lines:
        m = key_re.match(ln)
        if m:
            raw_k, raw_v = m.group(1), m.group(2)
            k = _norm_space(raw_k).rstrip(":")
            # fuzzy map for claims
            if "claims" in k.lower():
                k = "On-pack claims"
            # fuzzy map for packaging
            if k.lower().startswith("packaging"):
                k = "Packaging (primary color, secondary color, container type, closure type)"
            if k.lower().startswith("product name"):
                k = "Product name"
            if k.lower() == "category":
                k = "Category"
            if k.lower().startswith("type"):
                k = "Type/form"
            if k.lower() == "brand":
                k = "Brand"
            if k.lower().startswith("size"):
                k = "Size/volume"

            cur_key = k

            if cur_key == "On-pack claims":
                # claims can be inline, but we prefer bullets; capture inline too
                v = _clean_text(raw_v)
                if v:
                    # split by commas if present
                    parts = [p.strip() for p in v.split(",") if p.strip()]
                    out[cur_key] = parts
                else:
                    out[cur_key] = out.get(cur_key, [])
            else:
                out[cur_key] = _clean_text(raw_v) or out.get(cur_key, "")
            continue

        # claims bullets
        if cur_key == "On-pack claims":
            m2 = bullet_re.match(ln)
            if m2:
                c = _clean_text(m2.group(1).strip(" ,;"))
                if c:
                    out.setdefault("On-pack claims", [])
                    out["On-pack claims"].append(c)

    # Ensure claims list type
    if "On-pack claims" in out and not isinstance(out["On-pack claims"], list):
        out["On-pack claims"] = [str(out["On-pack claims"])]

    return out

def _dedup(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        key = it.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it.strip())
    return out

def _filter_claims_for_rank(claims: List[str], max_claims: int) -> List[str]:
    if not claims:
        return []
    claims = [_norm_space(c) for c in claims if c and c.lower() != "unknown"]
    claims = _dedup(claims)

    # Remove very long claims
    claims = [c[:80].rstrip() for c in claims if len(c) <= 200]

    # Remove noisy/hallucination-prone claims (best-effort)
    cleaned = []
    for c in claims:
        cl = c.lower()
        if any(re.search(p, cl) for p in NOISY_CLAIM_PATTERNS):
            continue
        cleaned.append(c)
    claims = cleaned

    # Prioritize keyword-bearing claims
    prioritized = []
    rest = []
    for c in claims:
        cl = c.lower()
        if any(k in cl for k in RANK_CLAIM_KEYWORDS):
            prioritized.append(c)
        else:
            rest.append(c)

    prioritized = _dedup(prioritized)
    rest = _dedup(rest)

    final = (prioritized + rest)[:max_claims]
    return final

def _short_packaging(packaging: str) -> str:
    """
    Compress packaging to a short tag to help ranking but reduce tokens.
    Example input: "dark grey, black, bottle, flip-top cap"
    Output: "bottle/flip-top"
    """
    p = _clean_text(packaging)
    if not p:
        return ""
    pl = p.lower()

    # detect container
    container = ""
    for c in ["pump bottle", "spray bottle", "aerosol can", "bottle", "tube", "jar", "pouch", "box"]:
        if c in pl:
            container = c.replace(" ", "")
            break

    closure = ""
    for cl in ["flip-top cap", "screw cap", "pump", "trigger spray", "spray nozzle", "aerosol"]:
        if cl in pl:
            closure = cl.split()[0] if cl in ["pump", "aerosol"] else cl.replace(" cap", "").replace(" ", "")
            break

    if container and closure:
        return f"{container}/{closure}"
    if container:
        return container
    return ""

def _short_size(size: str) -> str:
    """
    Keep compact size only (e.g., "13.5 fl oz", "400 mL").
    """
    s = _clean_text(size)
    if not s:
        return ""
    # keep first occurrence of oz/ml/g style token
    m = re.search(r"(\d+(?:\.\d+)?)\s*(fl\.?\s*oz|oz|ml|mL|g)\b", s, flags=re.I)
    if m:
        num = m.group(1)
        unit = m.group(2).replace(" ", "")
        return f"{num}{unit}"
    return s[:24]

def _short_brand(brand: str) -> str:
    b = _clean_text(brand)
    if not b:
        return ""
    # keep compact, remove weird punctuation spacing
    b = b.replace("  ", " ").strip()
    return b

def _short_name(name: str) -> str:
    n = _clean_text(name)
    if not n:
        return ""
    # avoid ultra-long names; keep first ~8 words
    words = n.split()
    if len(words) > 8:
        return " ".join(words[:8])
    return n

# --------- Main function ---------
def reduce_viu_for_reranker(
    viu_text: str,
    *,
    format: str = "one_line",
    max_claims: int = 3,
    include_brand: bool = True,
    include_category: bool = False,
    include_packaging_tag: bool = True,
    include_size: bool = True,
    keep_unknown: bool = False,
) -> str:
    """
    Reduce a verbose VIU to a short reranker-friendly string.

    Args:
      viu_text: bullet VIU string
      format: "one_line" (default) or "two_lines"
      max_claims: number of claims to keep (recommended 2-4)
      include_brand/category/packaging_tag/size: toggles
      keep_unknown: if True, keep literal "Unknown" (usually False for reranker)

    Returns:
      Reduced text string.
    """
    data = _parse_viu_bullets(viu_text)

    name = _short_name(str(data.get("Product name", "")))
    brand = _short_brand(str(data.get("Brand", "")))
    cat = _clean_text(str(data.get("Category", "")))
    typ = _clean_text(str(data.get("Type/form", "")))
    packaging = _short_packaging(str(data.get("Packaging (primary color, secondary color, container type, closure type)", "")))
    size = _short_size(str(data.get("Size/volume", "")))

    claims = data.get("On-pack claims", [])
    if isinstance(claims, str):
        claims = [claims]
    claims = _filter_claims_for_rank([str(c) for c in claims], max_claims=max_claims)

    # If we want to keep Unknown markers (generally not recommended for ranking)
    if keep_unknown:
        if not name:
            name = "Unknown"
        if include_brand and not brand:
            brand = "Unknown"
        if include_category and not cat:
            cat = "Unknown"
        if not typ:
            typ = "Unknown"
        if include_packaging_tag and not packaging:
            packaging = "Unknown"
        if include_size and not size:
            size = "Unknown"
        if not claims:
            claims = ["Unknown"]

    parts: List[str] = []

    # Build compact parts in a stable order
    if include_brand and brand:
        parts.append(brand)
    if name:
        parts.append(name)

    # Prefer Type/form for ranking; optionally add category
    if typ:
        parts.append(typ)
    if include_category and cat and cat.lower() != typ.lower():
        parts.append(cat)

    if include_size and size:
        parts.append(size)

    if include_packaging_tag and packaging:
        parts.append(packaging)

    if claims:
        parts.append(", ".join(claims))

    # Output formatting
    if format == "two_lines":
        head = " | ".join(parts[:4]) if len(parts) >= 4 else " | ".join(parts)
        tail = " | ".join(parts[4:]) if len(parts) > 4 else ""
        return head + (("\n" + tail) if tail else "")
    else:
        # default one_line
        return " | ".join(parts)

# --------- Example usage ---------
if __name__ == "__main__":
    sample_viu = """- Product name: Suave Moroccan Infusion
- Category: Shampoo
- Type/form: shampoo
- Brand: Suave
- Packaging (primary color, secondary color, container type, closure type): gold, gold, bottle, flip-top cap
- Size/volume: 32 fl oz
- On-pack claims:
  - Salon Perfect
  - Color Protect
  - Moroccan Argan Oil
  - Anti-Aging
  - Moisturizing
"""
    print(reduce_viu_for_reranker(sample_viu, max_claims=3))
