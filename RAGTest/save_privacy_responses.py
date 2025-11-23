"""
Utility để lưu responses được bảo vệ vào file
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

def save_privacy_response(
    question: str,
    original_response: str,
    protected_response: str,
    privacy_metadata: Dict[str, Any],
    output_dir: str = "./privacy_responses",
    format: str = "json"  # "json" or "txt"
) -> str:
    """
    Lưu response được bảo vệ vào file.
    
    Args:
        question: Câu hỏi gốc
        original_response: Response gốc từ LLM (chưa bảo vệ)
        protected_response: Response đã được bảo vệ
        privacy_metadata: Privacy metadata
        output_dir: Thư mục lưu file
        format: Format file ("json" hoặc "txt")
    
    Returns:
        Path đến file đã lưu
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "json":
        filename = f"privacy_response_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        data = {
            "timestamp": timestamp,
            "question": question,
            "original_response": original_response,
            "protected_response": protected_response,
            "privacy_metadata": privacy_metadata,
            "pii_detected": len(privacy_metadata.get("pii_entities", [])),
            "pii_density": privacy_metadata.get("pii_density", 0.0),
            "sentences_removed": privacy_metadata.get("eraser", {}).get("removed_count", 0),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    else:  # txt format
        filename = f"privacy_response_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Privacy-Protected Response\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Question: {question}\n\n")
            f.write(f"Original Response:\n{original_response}\n\n")
            f.write(f"Protected Response:\n{protected_response}\n\n")
            f.write("Privacy Metadata:\n")
            f.write(f"  - PII Detected: {len(privacy_metadata.get('pii_entities', []))}\n")
            f.write(f"  - PII Density: {privacy_metadata.get('pii_density', 0.0):.3f}\n")
            f.write(f"  - Sentences Removed: {privacy_metadata.get('eraser', {}).get('removed_count', 0)}\n")
            f.write(f"  - Average Risk: {privacy_metadata.get('eraser', {}).get('average_risk', 0.0):.3f}\n")
            f.write(f"  - Encrypted: {privacy_metadata.get('encryption', {}).get('enabled', False)}\n")
            f.write("\n" + "=" * 80 + "\n")
    
    return filepath


def save_privacy_responses_batch(
    responses: list,
    output_dir: str = "./privacy_responses",
    format: str = "json"
) -> str:
    """
    Lưu nhiều responses vào một file.
    
    Args:
        responses: List of dicts với keys: question, original_response, protected_response, privacy_metadata
        output_dir: Thư mục lưu file
        format: Format file
    
    Returns:
        Path đến file đã lưu
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "json":
        filename = f"privacy_responses_batch_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        data = {
            "timestamp": timestamp,
            "total_responses": len(responses),
            "responses": []
        }
        
        for resp in responses:
            data["responses"].append({
                "question": resp.get("question", ""),
                "original_response": resp.get("original_response", ""),
                "protected_response": resp.get("protected_response", ""),
                "privacy_metadata": resp.get("privacy_metadata", {}),
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    else:  # txt format
        filename = f"privacy_responses_batch_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Privacy-Protected Responses Batch\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Responses: {len(responses)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, resp in enumerate(responses, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"Response #{i}\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"Question: {resp.get('question', '')}\n\n")
                f.write(f"Original Response:\n{resp.get('original_response', '')}\n\n")
                f.write(f"Protected Response:\n{resp.get('protected_response', '')}\n\n")
                privacy_meta = resp.get('privacy_metadata', {})
                f.write("Privacy Stats:\n")
                f.write(f"  - PII Detected: {len(privacy_meta.get('pii_entities', []))}\n")
                f.write(f"  - PII Density: {privacy_meta.get('pii_density', 0.0):.3f}\n")
                f.write(f"  - Sentences Removed: {privacy_meta.get('eraser', {}).get('removed_count', 0)}\n")
                f.write("\n")
    
    return filepath

