"""
Script để kiểm tra Privacy Module có chạy và xem logs
"""
import logging
import sys
from config import Config
from privacy import apply_privacy_to_response

# Setup logging để xem privacy logs
logging.basicConfig(
    level=logging.DEBUG,  # Hiển thị tất cả logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('privacy_logs.txt', encoding='utf-8')
    ]
)

# Enable privacy module logging
logger = logging.getLogger('privacy.privacy_summary')
logger.setLevel(logging.DEBUG)

print("=" * 80)
print("Privacy Module Checker")
print("=" * 80)

# Load config
cfg = Config()

# Check privacy config
print("\n[1] Privacy Configuration:")
if hasattr(cfg, 'privacy'):
    privacy_cfg = cfg.privacy
    print(f"  enable_privacy_summary: {privacy_cfg.get('enable_privacy_summary', False)}")
    print(f"  presidio_language: {privacy_cfg.get('presidio_language', 'en')}")
    print(f"  privacy_log_stats: {privacy_cfg.get('privacy_log_stats', False)}")
else:
    print("  ⚠️  Privacy config not found!")

# Test privacy module với sample response
print("\n[2] Testing Privacy Module:")

# Mock response object
class MockResponse:
    def __init__(self, text):
        self.response = text
        self.source_nodes = []
        self.metadata = {}

# Test với text có PII
test_text = "John Doe's email is john.doe@example.com and his phone is +1-555-123-4567."
test_response = MockResponse(test_text)

print(f"  Original text: {test_text}")

try:
    processed_response, privacy_metadata = apply_privacy_to_response(
        test_response, "test query", cfg
    )
    
    print(f"  ✓ Privacy module executed successfully!")
    print(f"  Processed text: {processed_response.response[:100]}...")
    
    print("\n[3] Privacy Metadata:")
    print(f"  PII entities detected: {len(privacy_metadata.get('pii_entities', []))}")
    for entity in privacy_metadata.get('pii_entities', [])[:5]:
        print(f"    - {entity.get('entity_type')}: score={entity.get('score', 'N/A')}")
    
    print(f"  PII density: {privacy_metadata.get('pii_density', 0.0):.3f}")
    print(f"  Sentences removed: {privacy_metadata.get('eraser', {}).get('removed_count', 0)}")
    print(f"  Average risk: {privacy_metadata.get('eraser', {}).get('average_risk', 0.0):.3f}")
    print(f"  Encrypted: {privacy_metadata.get('encryption', {}).get('enabled', False)}")
    
    print("\n[4] Log Files:")
    print("  - Console output (above)")
    print("  - File: privacy_logs.txt (created in current directory)")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Check complete! Logs saved to: privacy_logs.txt")
print("=" * 80)

