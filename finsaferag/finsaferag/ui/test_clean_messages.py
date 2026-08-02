"""
Quick test to verify HTML cleaning works
"""
import re
import html

def _clean_html_text(text: str) -> str:
    """Remove HTML tags/artifacts and return cleaned plain text (may be empty)."""
    if not text:
        return ""
    
    raw = str(text)
    
    # Step 1: Remove ALL HTML tags (multiple passes to catch nested)
    for _ in range(3):  # Multiple passes for nested tags
        raw = re.sub(r'<[^>]*>', '', raw)
    
    # Step 2: Remove specific problematic patterns
    patterns_to_remove = [
        "</div>", "<div>", "<div ", "</div ", 
        "<div class=\"message-time\">", "<div class='message-time'>",
        "<div class=\"message-time\"", "<div class='message-time'",
        "<span>", "</span>", "<p>", "</p>",
        "<br>", "<br/>", "<br />",
    ]
    for pattern in patterns_to_remove:
        raw = raw.replace(pattern, "")
    
    # Step 3: Decode HTML entities
    raw = raw.replace("&lt;", "").replace("&gt;", "")
    raw = raw.replace("&amp;", "&").replace("&nbsp;", " ")
    raw = raw.replace("&#96;", "`")
    
    # Step 4: Try html.unescape to catch any remaining entities
    try:
        raw = html.unescape(raw)
    except:
        pass
    
    # Step 5: Clean whitespace
    raw = " ".join(raw.split())  # Normalize whitespace
    raw = raw.strip()
    
    return raw


print("=" * 70)
print("HTML CLEANING TEST - Verify </div> removal")
print("=" * 70)

test_cases = [
    ("</div>", "Should be empty"),
    ("<div>Hello</div>", "Should be: Hello"),
    ("CEO của Apple</div>", "Should be: CEO của Apple"),
    ("Answer: <div class='test'>Tim Cook</div>", "Should be: Answer: Tim Cook"),
    ("Biên lợi nhuận của Tesla", "Should be unchanged"),
    ("<div><div>Nested</div></div>", "Should be: Nested"),
    ("Text &lt;div&gt; more text", "Should remove escaped HTML"),
]

print()
all_pass = True

for test_input, expected in test_cases:
    cleaned = _clean_html_text(test_input)
    
    # Check if any HTML artifacts remain
    has_html = "</div>" in cleaned or "<div" in cleaned.lower()
    
    status = "✅ PASS" if not has_html else "❌ FAIL"
    if has_html:
        all_pass = False
    
    print(f"{status}")
    print(f"  Input:    '{test_input}'")
    print(f"  Output:   '{cleaned}'")
    print(f"  Expected: {expected}")
    print()

print("=" * 70)
if all_pass:
    print("✅ ALL TESTS PASSED! HTML cleaning works correctly.")
else:
    print("❌ SOME TESTS FAILED! Review the cleaning function.")
print("=" * 70)

