# Chạy Relation Network trên Google Colab

## Bước 1 – Bật GPU

`Runtime → Change runtime type → T4 GPU`

---

## Bước 2 – Clone repo và cài thư viện

```bash
# Clone repo của bạn (thay bằng URL repo của bạn)
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /content/RelationNet
%cd /content/RelationNet/miniimagenet

# Cài thư viện
!pip install -q scipy Pillow
```

---

## Bước 3 – Chuẩn bị dữ liệu

### Cách A – Tải từ Google Drive (khuyến nghị)
```bash
from google.colab import drive
drive.mount('/content/drive')

# Giải nén data từ Drive vào /content/
!unzip -q "/content/drive/MyDrive/datasets/miniImagenet.zip" -d /content/data/
```

### Cách B – Tải bằng gdown (nếu có link Google Drive)
```bash
!pip install -q gdown
!gdown --id YOUR_FILE_ID -O /content/miniImagenet.zip
!unzip -q /content/miniImagenet.zip -d /content/data/
```

Cấu trúc thư mục data cần có:
```
/content/data/miniImagenet/
    images/          ← (nếu dùng Format 2 JSON)
        n01532829/
        n01558993/
        ...
    train/           ← (nếu dùng cách quét thư mục gốc)
    val/
```

---

## Bước 4 – Chạy training

### Backbone Conv4 (mặc định, ảnh 84×84)
```bash
!python miniimagenet_train_one_shot.py \
    -w 5 -s 1 -b 15 \
    --backbone conv4 \
    --split_file /content/RelationNet/miniimagenet/split_example.json \
    --data_root  /content/data/miniImagenet/images \
    --model_dir  /content/drive/MyDrive/RelationNet/models \
    --log_dir    /content/drive/MyDrive/RelationNet/logs
```

### Backbone ResNet50 (pretrained ImageNet, ảnh 224×224)
> ⚠️ **Lưu ý VRAM**: ResNet50 + 224×224 tốn nhiều bộ nhớ. Trên Colab T4 (~15GB), giảm `-b` xuống 5~8.
```bash
!python miniimagenet_train_one_shot.py \
    -w 5 -s 1 -b 5 \
    --backbone   resnet50 \
    --image_size 224 \
    --split_file /content/RelationNet/miniimagenet/split_example.json \
    --data_root  /content/data/miniImagenet/images \
    --model_dir  /content/drive/MyDrive/RelationNet/models \
    --log_dir    /content/drive/MyDrive/RelationNet/logs
```

### 5-shot với ResNet50
```bash
!python miniimagenet_train_few_shot.py \
    -w 5 -s 5 -b 5 \
    --backbone   resnet50 \
    --image_size 224 \
    --split_file /content/RelationNet/miniimagenet/split_example.json \
    --data_root  /content/data/miniImagenet/images \
    --model_dir  /content/drive/MyDrive/RelationNet/models \
    --log_dir    /content/drive/MyDrive/RelationNet/logs
```

### Dùng JSON split Format 2 (mỗi split 1 file, kiểu FEAT/Chen et al.)
```bash
!python miniimagenet_train_one_shot.py \
    -w 5 -s 1 -b 10 \
    --backbone   resnet50 \
    --train_json /content/data/splits/train.json \
    --test_json  /content/data/splits/val.json \
    --data_root  /content/data/miniImagenet/images \
    --model_dir  /content/drive/MyDrive/RelationNet/models \
    --log_dir    /content/drive/MyDrive/RelationNet/logs
```

---

## Tham số quan trọng

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `-w` | 5 | N-way (số class/episode) |
| `-s` | 1/5 | K-shot (số mẫu support/class) |
| `-b` | 15 | Số query mỗi class mỗi episode |
| `-e` | 500000 | Tổng số episode |
| `-l` | 0.001 | Learning rate |
| `--backbone` | `conv4` | Backbone: `conv4` hoặc `resnet50` |
| `--image_size` | auto | Kích thước ảnh (tự động: 84 cho conv4, 224 cho resnet50) |
| `--pretrained` | True | Dùng trọng số ImageNet cho resnet50 |
| `--model_dir` | `./models` | Nơi lưu checkpoint |
| `--log_dir` | `./logs` | Nơi lưu log CSV |
| `--split_file` | None | File JSON split (Format 1) |
| `--train_json` | None | File JSON train (Format 2) |
| `--test_json` | None | File JSON val/test (Format 2) |
| `--data_root` | None | Thư mục gốc chứa ảnh |
| `--train_key` | `train` | Key meta-train trong split_file |
| `--test_key` | `val` | Key meta-test trong split_file |

---

## Lưu log và model vào Drive để không mất sau khi session kết thúc

```bash
# Mount Drive trước (xem Bước 3)
--model_dir /content/drive/MyDrive/RelationNet/models
--log_dir   /content/drive/MyDrive/RelationNet/logs
```

## Cập nhật code mới từ repo

```bash
%cd /content/RelationNet
!git pull origin main
```
