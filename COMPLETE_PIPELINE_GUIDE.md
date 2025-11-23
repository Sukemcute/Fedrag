# Hướng Dẫn Toàn Bộ Pipeline FedE4RAG: Từ Upstream Đến Downstream

## 📋 Tổng Quan

FedE4RAG là một framework sử dụng Federated Learning để fine-tune embedding models cho RAG (Retrieval-Augmented Generation) systems. Pipeline bao gồm 2 phần chính:

1. **Upstream (Federated Learning Training)**: Fine-tune embedding models sử dụng federated learning
2. **Downstream (RAG Evaluation)**: Đánh giá các embedding models đã fine-tune trong RAG pipeline

---

## ⚡ Tóm Tắt Nhanh Các Bước

### Upstream (Federated Learning)
```bash
# 1. Cài Miniconda và tạo environment
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n fedrag python=3.11 -y
conda activate fedrag

# 2. Clone repo và cài dependencies
git clone https://github.com/Sukemcute/Fedrag.git
cd Fedrag/FedE
pip install -r requirements.txt
pip install transformers==4.35.0
pip install "numpy<2"

# 3. Chạy training
bash run.sh

# 4. Model sẽ được lưu tại:
# - Model cuối: Fedrag/FedE/x-model_*.bin
# - Models từng round: Fedrag/FedE/checkpoints/
```

### Downstream (RAG Evaluation)
```bash
# 1. Cài dependencies
cd Fedrag/RAGTest
pip install -r requirements.txt
pip install openai==1.55.3
pip install jury --no-deps
pip install gdown
pip install -U bitsandbytes

# 2. Download data và config
cd data
gdown https://drive.google.com/uc?id=1uiC3TfaUgbydukAAUgI9QR_Nj34WDIct -O test_corpus_backup.json
cd ..
rm config.toml
gdown https://drive.google.com/uc?id=1d-rlvn0IHeG9NRt-KRKgrhdssFG_GCES -O config.toml

# 3. Convert model (nếu cần)
cd embs
python change.py ../../FedE/x-model_*.bin

# 4. Chạy evaluation
cd ..
python main_100_test.py --model="../FedE/x-model_*_converted"
```

---

## 🔄 Pipeline Tổng Quan

```
┌─────────────────────────────────────────────────────────────────┐
│                    UPSTREAM PIPELINE                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │ 1. Chuẩn bị  │ -> │ 2. Training  │ -> │ 3. Save Model│    │
│  │   Data       │    │   (FL)        │    │   (.bin)     │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL CONVERSION                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │ 4. Convert   │ -> │ 5. Save      │ -> │ 6. Ready for │    │
│  │   .bin -> HF │    │   HF Format  │    │   Downstream │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DOWNSTREAM PIPELINE                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │ 7. Load      │ -> │ 8. Build     │ -> │ 9. Evaluate  │    │
│  │   Model      │    │   Index      │    │   RAG System  │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 PHẦN 1: UPSTREAM - FEDERATED LEARNING TRAINING

### Mục Đích
Fine-tune embedding models (như BGE-base-en) sử dụng federated learning với dữ liệu phân tán trên nhiều clients.

### Bước 1: Cài Đặt Miniconda

```bash
# Tải và cài đặt Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 

bash Miniconda3-latest-Linux-x86_64.sh

# Kích hoạt Conda
source ~/.bashrc

# Kiểm tra phiên bản Conda
conda --version
```

### Bước 2: Tạo Conda Environment

```bash
# Chấp nhận terms of service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Tạo environment mới
conda create -n fedrag python=3.11 -y

# Khởi tạo conda
conda init bash
source ~/.bashrc

# Kích hoạt environment
conda activate fedrag
```

### Bước 3: Clone Repository và Cài Đặt Dependencies

```bash
# Clone repository
git clone https://github.com/Sukemcute/Fedrag.git

# Di chuyển vào thư mục FedE
cd Fedrag/FedE

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt transformers với version cụ thể
pip install transformers==4.35.0

# Cài đặt numpy (version < 2)
pip install "numpy<2"
```

**Yêu Cầu Phần Cứng:**
- GPU với ít nhất 80GB memory (khuyến nghị: A40 hoặc tương đương)
- Batch size 16 yêu cầu GPU memory cao

### Bước 4: Chuẩn Bị Dữ Liệu Training

#### 4.1. Download Dataset
Dataset training có sẵn tại: [DocAILab/FedE4RAG_Dataset - train_data](https://huggingface.co/datasets/DocAILab/FedE4RAG_Dataset/tree/main/FEDE4FIN)

#### 4.2. Chọn Dữ Liệu Training
Chỉnh sửa file `Fedrag/FedE/select_data.json` để chọn dữ liệu training:
Sửa trong `Fedrag/FedE/flgo/benchmark/fedrag_classification/core.py`  -> sửa thành file dataset khác là được.

```json
{
  "data_path": "/path/to/train_data/data_10000_random.json"
}
```

**Các file training có sẵn:**
- `data_1000_random.json`
- `data_2000_random.json`
- `data_5000_random.json`
- `data_10000_random.json`
- `data_20000_random.json`
- `data_50000_random.json`

### Bước 5: Cấu Hình Training
(Cần chỉnh thông số thì cấu hình vào đây, không thì để nguyên. )
Chỉnh sửa file `Fedrag/FedE/main.py`:

```python
import flgo
import flgo.algorithm.fedrag as fedrag
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 1. Đặt tên task (thư mục lưu kết quả)
task = './num5_alpha05'

# 2. Cấu hình benchmark và partitioner
config = {
    'benchmark': {'name': 'flgo.benchmark.fedrag_classification'},
    'partitioner': {
        'name': 'IDPartitioner', 
        'para': {'num_clients': 5}  # Số lượng clients
    }
}

# 3. Generate task nếu chưa tồn tại
if not os.path.exists(task): 
    flgo.gen_task(config, task_path=task)

# 4. Khởi tạo và chạy federated learning
fedavg_runner = flgo.init(
    task=task, 
    algorithm=fedrag,
    option={
        'num_rounds': 25,      # Số rounds federated learning
        'num_epochs': 1,        # Số epochs mỗi round
        'gpu': 0,              # GPU ID
        'batch_size': 8,       # Batch size
        'learning_rate': 0.00001  # Learning rate
    }
)
fedavg_runner.run()
```

**Các Tham Số Quan Trọng:**
- `num_rounds`: Số vòng federated learning (khuyến nghị: 20-30)
- `num_clients`: Số lượng clients (ví dụ: 5)
- `batch_size`: Batch size (phụ thuộc vào GPU memory)
- `learning_rate`: Learning rate (khuyến nghị: 1e-5)

### Bước 6: Chạy Training

```bash
# Đảm bảo đang ở trong thư mục Fedrag/FedE và đã activate conda environment
conda activate fedrag
cd Fedrag/FedE

# Chạy training
bash run.sh
```

**Lưu Ý về `run.sh`:**
- File `run.sh` có thể được cấu hình cho SLURM hoặc chạy trực tiếp
- Nếu không dùng SLURM, có thể chạy trực tiếp: `python main.py`

### Bước 7: Kết Quả Training

Sau khi training xong, model sẽ được lưu tự động:

**Output Files:**
- **Model cuối cùng**: `Fedrag/FedE/x-model_{timestamp}.bin` 
  - Model này được lưu sau khi hoàn thành tất cả các rounds
- **Models từng round**: `Fedrag/FedE/checkpoints/`
  - Mỗi model sau mỗi round sẽ được lưu trong thư mục `checkpoints`
  - Training thường chạy qua 25 rounds → sẽ có 25 models trong checkpoints

**Ví dụ cấu trúc:**
```
Fedrag/FedE/
  ├── x-model_2025-11-23_04-01-02.bin  ← Model cuối cùng (sau 25 rounds)
  ├── checkpoints/
  │   ├── round_0_model.bin
  │   ├── round_1_model.bin
  │   ├── ...
  │   └── round_24_model.bin
  └── logs/
      └── ...
```

**Lưu Ý:**
- Model được lưu dưới dạng `.bin` file chứa `state_dict`
- Format: `{'model.embedding.weight': tensor(...), ...}`
- Model cuối cùng (bên ngoài checkpoints) là model tốt nhất sau khi hoàn thành tất cả rounds
- Cần convert sang HuggingFace format để sử dụng trong downstream


---

## 📊 PHẦN 3: DOWNSTREAM - RAG EVALUATION

### Bước 8: Chuẩn Bị Môi Trường Downstream

```bash
# Di chuyển vào thư mục RAGTest
cd Fedrag/RAGTest

# Cài đặt dependencies
pip install -r requirements.txt
pip install openai==1.55.3
pip install jury --no-deps
pip install gdown  # Để download files từ Google Drive
pip install -U bitsandbytes  # Cho quantization models
```

### Bước 9: Convert Model

### Mục Đích
Convert model từ format `.bin` (state_dict) sang HuggingFace format để sử dụng trong downstream.


Sử dụng script `RAGTest/embs/change.py`:

```bash
cd RAGTest/embs

# Cách 1: Convert từ file .bin  (dùng cái này)
python change.py ../FedE/x-model_2025-11-23_04-01-02.bin

# Cách 2: Convert từ directory chứa .bin/.pt file
python change.py /path/to/model/directory

# Cách 3: Chỉ định output directory
python change.py ../FedE/x-model.bin ./converted_model

# Cách 4: Chỉ định base model khác (nếu dùng BGE-large)
python change.py ../FedE/x-model.bin ./converted_model "BAAI/bge-large-en-v1.5"
```

**Script sẽ:**
1. Load `state_dict` từ file `.bin`
2. Xóa prefix `'model.'` từ keys (nếu có)
3. Load base model (mặc định: `BAAI/bge-base-en`) để lấy config và tokenizer
4. Load `state_dict` vào model
5. Save model dưới dạng HuggingFace format (có thể load bằng `from_pretrained`)

**Output:**
```
converted_model/
  ├── config.json
  ├── pytorch_model.bin (hoặc model.safetensors)
  ├── tokenizer_config.json
  ├── vocab.txt
  └── ...
```

**Lưu Ý:**
- Base model phải khớp với model đã train (ví dụ: nếu train từ `BAAI/bge-base-en` thì dùng `BAAI/bge-base-en` làm base)
- Nếu train từ `BAAI/bge-large-en-v1.5`, chỉ định base model tương ứng


### Bước 10: Download Dữ Liệu và Config

```bash
# Di chuyển vào thư mục data
cd data

# Download test corpus từ Google Drive
gdown https://drive.google.com/uc?id=1uiC3TfaUgbydukAAUgI9QR_Nj34WDIct -O test_corpus_backup.json

# Quay lại thư mục RAGTest
cd ..

# Xóa file config.toml cũ (nếu có)
rm config.toml 

# Download config.toml mới từ Google Drive
gdown https://drive.google.com/uc?id=1d-rlvn0IHeG9NRt-KRKgrhdssFG_GCES -O config.toml
```

**Lưu Ý:**
- File `test_corpus_backup.json` sẽ được lưu trong `RAGTest/data/`
- File `config.toml` sẽ được lưu trong `RAGTest/`
- Đảm bảo đã cài `gdown` để download từ Google Drive

### Bước 11: Cấu Hình Downstream (Tùy Chọn)

Nếu cần chỉnh sửa, mở file `RAGTest/config.toml`:

```toml
[api_keys]
api_key = "sk-your-openai-api-key-here"  # Nếu dùng OpenAI API
api_base = "https://api.openai.com/v1"
api_name = "gpt-4o-mini"  # hoặc "deepseek-r1:7b", "llama", etc.
auth_token = ""  # HuggingFace token nếu cần

[settings]
llm = "gpt-4o-mini"  # LLM để generate response
embeddings = ""  # Để trống, sẽ dùng --model argument
split_type = "sentence"  # "sentence" hoặc "word"
chunk_size = 2048
dataset = "json_download"  # Tên dataset
source_dir = "../wiki"  # Thư mục chứa documents
persist_dir = "storage"  # Thư mục lưu index
retriever = "Vector"  # "Vector", "BM25", "Tree", etc.
postprocess_rerank = "long_context_reorder"
query_transform = "none"
n = 100  # Số lượng test samples
llamaIndexEvaluateModel = "Qwen/Qwen1.5-7B-Chat"  # Eval model (optional)
deepEvalEvaluateModel = "Qwen/Qwen1.5-7B-Chat"   # Eval model (optional)
```

**Lưu Ý:**
- File `config.toml` đã được download từ Google Drive, thường đã được cấu hình sẵn
- Chỉ cần chỉnh sửa nếu muốn thay đổi các tham số

### Bước 12: Chạy Evaluation
Chạy test cả 3 trường hợp để so sánh việc dùng model embedding có sẵn (BAAI/bge-base-en) và dùng model embedding đã được train ở Upstream.

**Sự khác biệt giữa `main_100_test.py` và `main_response.py`?**
->  `main_100_test.py` chỉ test retrieval, `main_response.py` test cả response generation với NLG metrics.

#### Trường Hợp 1: Test với Pretrained Model (Baseline)

```bash
# Đảm bảo đang ở trong thư mục RAGTest
cd Fedrag/RAGTest

# Test với BGE-base-en (pretrained)
python main_100_test.py --model="BAAI/bge-base-en"

# Test với BGE-large
python main_100_test.py --model="BAAI/bge-large-en-v1.5"
```

#### Trường Hợp 2: Test với Fine-tuned Model (Từ Upstream)

**Bước 2.1: Convert Model (Nếu chưa convert)**

```bash
# Convert model từ upstream sang HuggingFace format
cd Fedrag/RAGTest/embs
python change.py ../../FedE/x-model_2025-11-23_04-01-02.bin
```

**Bước 2.2: Test với Model Đã Convert**

```bash
# Quay lại thư mục RAGTest
cd Fedrag/RAGTest

# Test với model đã convert từ upstream
python main_100_test.py --model="../FedE/x-model_2025-11-23_04-01-02_converted"

# Hoặc nếu đã convert vào thư mục khác
python main_100_test.py --model="./converted_model"
```

#### Trường Hợp 3: Test với Response Generation

```bash
# Test với response generation (bao gồm cả NLG metrics)
cd Fedrag/RAGTest
python main_response.py --model="../FedE/x-model_2025-11-23_04-01-02_converted"
```

#### Trường Hợp 4: Batch Testing (Nhiều Models)  (không cần test cái này)

Sửa file `Fedrag/RAGTest/bash.sh` hoặc `Fedrag/RAGTest/bash1.sh`:

```bash
# bash.sh - Test retrieval metrics
python main_100_test.py --model="/path/to/model1" > log1.log
python main_100_test.py --model="/path/to/model2" > log2.log

# bash1.sh - Test với response generation
python main_response.py --model="/path/to/model1" > log1.log
python main_response.py --model="/path/to/model2" > log2.log
```

Chạy:
```bash
cd Fedrag/RAGTest
bash bash.sh      # Chỉ test retrieval
bash bash1.sh     # Test với response generation
```

### Bước 13: Kết Quả Evaluation

**Output Files:**

1. **`0407_0318+0322.txt`** (hoặc tương tự): File chứa kết quả evaluation
   - TRT metrics (Hit@k, Recall@k, Precision@k, F1, EM, MRR, MAP, NDCG)
   - NLG metrics averages (ROUGE, METEOR, CHRF, WER, CER, Perplexity)
   - Metrics theo categories (domain-relevant, metrics-generated, novel-generated)

2. **`storage-{dataset}-{model}-{config}/`**: Thư mục chứa vector index
   - Có thể tái sử dụng để tránh rebuild index

3. **Log files** (khi dùng bash scripts):
   - `logs_test_100/`: Thư mục chứa log files

**Ví dụ Output:**
```
TRT ----------------------------------------------------------------------------------------------------
n: 50
F1: 0.6666666666666667
em: 0.6
mrr: 0.7933333333333334
hit1: 0.82
hit10: 0.88
MAP: 0.78
NDCG: 0.815355919212455

NLG Evaluation Metrics Averages:
cos_1: 0.7200
recall_1: 0.6167
precision: 0.7200
chrf_pp: 0.2243
perplexity: 178.7065
rouge_rouge1: 0.2297
rouge_rouge2: 0.1071
...

NLG Evaluation Metrics Averages by Category:
--- Category: domain-relevant (Evaluations: 17) ---
  hit_1: 0.5294
  rouge_rouge1: 0.3365
  ...
```

---

## 📈 Các Metrics Được Tính

### Retrieval Metrics (TRT - Text Retrieval Task)
- **Hit@k**: Hit rate tại top-k (k=1,3,5,10)
- **Recall@k**: Recall tại top-k
- **Precision@k**: Precision tại top-k
- **F1**: F1 score
- **EM**: Exact Match
- **MRR**: Mean Reciprocal Rank
- **MAP**: Mean Average Precision
- **NDCG**: Normalized Discounted Cumulative Gain

### Generation Metrics (NLG - Natural Language Generation)
- **ROUGE**: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum (với P, R)
- **METEOR**: METEOR score
- **CHRF/CHRF++**: CHRF score
- **WER/CER**: Word/Character Error Rate
- **Perplexity**: Perplexity
- **Cosine Similarity**: cos_1, cos_3, cos_5, cos_10

### Categories
Kết quả được phân loại theo `question_type`:
- **domain-relevant**: Câu hỏi về domain cụ thể
- **metrics-generated**: Câu hỏi về số liệu/metrics
- **novel-generated**: Câu hỏi mới/tổng hợp

---

## 🐛 Troubleshooting

### Upstream Issues

**Lỗi: Out of Memory**
- Giảm `batch_size` trong `main.py`
- Sử dụng GPU có memory lớn hơn
- Giảm số lượng clients

**Lỗi: Model không save**
- Kiểm tra path trong `fedrag.py` (dòng 45, 56)
- Đảm bảo có quyền ghi vào thư mục

### Conversion Issues

**Lỗi: FileNotFoundError**
- Kiểm tra đường dẫn model file
- Đảm bảo file `.bin` tồn tại
- Kiểm tra quyền đọc file

**Lỗi: Missing keys khi load state_dict**
- Thường là bình thường (một số keys có thể missing)
- Kiểm tra base model có khớp với model đã train không

### Downstream Issues

**Lỗi: Model không load được**
- Kiểm tra đường dẫn model
- Kiểm tra HuggingFace token nếu dùng private model
- Kiểm tra GPU memory

**Lỗi: Device mismatch (CPU vs CUDA)**
- Đảm bảo embedding model và LLM đều trên cùng device
- Xem `RAGTest/embs/embedding.py` và `RAGTest/llms/huggingface_model.py`


**Lỗi: Dataset không tìm thấy**
- Kiểm tra `dataset` name trong config
- Kiểm tra file trong `data/` folder
- Kiểm tra `source_dir` path

---

## 📝 Checklist Hoàn Chỉnh

### Upstream
- [ ] Cài đặt Miniconda
- [ ] Tạo conda environment `fedrag` với Python 3.11
- [ ] Clone repository `https://github.com/Sukemcute/Fedrag.git`
- [ ] Cài đặt dependencies (`requirements.txt`, `transformers==4.35.0`, `numpy<2`)
- [ ] Download training dataset
- [ ] Chỉnh sửa `select_data.json` (nếu cần)
- [ ] Cấu hình `main.py` (num_rounds, num_clients, batch_size, learning_rate)
- [ ] Chạy training (`bash run.sh`)
- [ ] Kiểm tra model output:
  - [ ] Model cuối cùng: `Fedrag/FedE/x-model_*.bin`
  - [ ] Models từng round: `Fedrag/FedE/checkpoints/`

### Conversion
- [ ] Xác định base model (BGE-base hoặc BGE-large)
- [ ] Chạy `change.py` để convert model từ `.bin` sang HuggingFace format
- [ ] Kiểm tra output directory có đầy đủ files (config.json, pytorch_model.bin, tokenizer files)

### Downstream
- [ ] Cài đặt dependencies (`requirements.txt`, `openai==1.55.3`, `jury`, `gdown`, `bitsandbytes`)
- [ ] Download `test_corpus_backup.json` từ Google Drive vào `RAGTest/data/`
- [ ] Download `config.toml` từ Google Drive vào `RAGTest/`
- [ ] Test với pretrained model (baseline)
- [ ] Convert model từ upstream (nếu chưa convert)
- [ ] Test với fine-tuned model
- [ ] So sánh kết quả
- [ ] Phân tích metrics theo categories

---

## 🚀 Next Steps

1. **So Sánh Models**: Test nhiều models và so sánh kết quả
2. **Fine-tune Hyperparameters**: Điều chỉnh `chunk_size`, `retriever`, `postprocess_rerank`
3. **Enable Advanced Metrics**: Uncomment các metrics trong code (Llama_, DeepEval_, UpTrain_)
4. **Custom Evaluation**: Tạo custom metrics hoặc tích hợp frameworks khác
5. **Export Results**: Export kết quả sang CSV/JSON để phân tích

---

## 📚 Tài Liệu Tham Khảo

- **README.md**: Tổng quan về project
- **DOWNSTREAM_GUIDE.md**: Hướng dẫn chi tiết downstream
- **config.toml**: Tất cả các tham số cấu hình
- **Dataset**: [DocAILab/FedE4RAG_Dataset](https://huggingface.co/datasets/DocAILab/FedE4RAG_Dataset)

---

*Cập nhật lần cuối: 2025-01-XX*

