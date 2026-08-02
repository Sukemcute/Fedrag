"""
Test script to verify HTML cleaning works correctly
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


# Test cases
test_cases = [
    "</div>",
    "<div>Hello</div>",
    "Answer: <div>Test</div>",
    "Normal text",
    "</div></div></div>",
    "<div class='message-time'>Time</div>Answer",
    "&lt;div&gt;Escaped&lt;/div&gt;",
    "CEO của Apple là ai",
    "Biên lợi nhuận của Tesla",
]

print("=" * 60)
print("HTML CLEANING TEST")
print("=" * 60)

for test in test_cases:
    cleaned = _clean_html_text(test)
    status = "✅ CLEAN" if cleaned else "❌ EMPTY"
    print(f"\n{status}")
    print(f"Input:  '{test}'")
    print(f"Output: '{cleaned}'")
    print("-" * 60)

print("\n✅ Test completed!")

