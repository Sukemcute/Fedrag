#control randomness from t5 infill (make sure the output is unique in every examples, but still keeping the model semantic)
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, T5Tokenizer, T5ForConditionalGeneration
import torch, re

#test lai full pipeline ner va t5 infill

# =========================================================
# 1️⃣ Load NER model
# =========================================================
MODEL_PATH = "RAGTest/bert_final_model"
tokenizer_ner = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model_ner = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

ner_pipeline = pipeline(
    "token-classification",
    model=model_ner,
    tokenizer=tokenizer_ner,
    aggregation_strategy="simple",
    device=0  # GPU; bỏ nếu chạy CPU
)

# =========================================================
# 2️⃣ Mask text theo NER entities
# =========================================================
def mask_text(text, ents):
    ents_sorted = sorted(ents, key=lambda x: x['start'])
    masked_text = ""
    last_idx = 0

    for ent in ents_sorted:
        start, end = ent['start'], ent['end']
        label = ent['entity_group'].upper()
        placeholder = f"[{label}]"  # chỉ giữ label, bỏ đánh số
        masked_text += text[last_idx:start] + placeholder
        last_idx = end

    masked_text += text[last_idx:]
    return masked_text

# =========================================================
# 3️⃣ Load T5 PII-infill model
# =========================================================
t5_model_dir = "RAGTest/t5_infill"
tok_t5 = T5Tokenizer.from_pretrained(t5_model_dir)
model_t5 = T5ForConditionalGeneration.from_pretrained(t5_model_dir)

def infill_pii(masked_text, model, tok, max_len_in=128, max_len_out=96):
    TASK_PREFIX = "infill pii: "
    text = TASK_PREFIX + masked_text.strip()
    inputs = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=max_len_in)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=1.3,
            top_k=50,
            top_p=0.9,
            max_new_tokens=max_len_out,
        )
    return tok.decode(outputs[0], skip_special_tokens=True)


# =========================================================
# 4️⃣ Hàm thay placeholder bằng text sinh ra
# =========================================================
def post_infill(masked_text, infilled_text):
    clean_infilled = re.sub(r"\[[A-Z_0-9]+\]", "", infilled_text).strip()
    placeholders = re.findall(r"\[[A-Z_0-9]+\]", masked_text)

    if len(placeholders) == 1:
        final_text = masked_text.replace(placeholders[0], clean_infilled)
    else:
        pairs = re.findall(r"(\[[A-Z_0-9]+\])\s*([^\[]+)", infilled_text)
        final_text = masked_text
        if pairs:
            for tag, val in pairs:
                final_text = final_text.replace(tag, val.strip())
        else:
            parts = clean_infilled.split()
            for i, ph in enumerate(placeholders):
                value = parts[i] if i < len(parts) else ""
                final_text = final_text.replace(ph, value)
    return " ".join(final_text.split())

# =========================================================
# 5️⃣ Loop infill cho đến khi không còn mask
# =========================================================
def loop_infill(masked_text, model, tok, max_loops=100):
    text = masked_text
    for _ in range(max_loops):
        placeholders = re.findall(r"\[[A-Z_0-9]+\]", text)
        if not placeholders:
            break
        infilled_text = infill_pii(text, model, tok)
        text = post_infill(text, infilled_text)
    return text

# =========================================================
# 6️⃣ Full pipeline
# =========================================================
def generate_pii_text(text):
    ents = ner_pipeline(text)
    masked = mask_text(text, ents)
    final_text = loop_infill(masked, model_t5, tok_t5)
    return masked, final_text

# =========================================================
# 7️⃣ Test
# =========================================================
input_text = """
         Subject: Group Messaging for Admissions Process

Good morning, everyone,

alibek.akbasov@gmail.com works at OpenAI, my phone number is +1-202-555-0173. Let's schedule a meeting with the following people:
- wynqvrh053 - Meeting at 10:20am
- luka.burg - Meeting at 21/12/2025
- qahil.wittauer - Meeting at quarter past 13
- gholamhossein.ruschke - Meeting at 9:47 PM
- pdmjrsyoz1460 are also invited. Tran Thi Thuy Hang will join us too.
"""

masked_text, _ = generate_pii_text(input_text)
print("🧩 Masked text:", masked_text, "\n")

for i in range(1):
    _, final_text = generate_pii_text(input_text)
    print(f"✅ Run {i+1}: {final_text}\n")
