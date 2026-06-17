# Tools & Utilities

Folder chứa các công cụ hỗ trợ cho data preprocessing.

## 📋 Danh sách tools:

### 1. `clean_preprocessed.py`
**Mục đích**: Xóa dữ liệu đã preprocessing để tạo lại từ đầu

**Sử dụng**:
```bash
cd ..
python tools/clean_preprocessed.py
```

**Khi nào dùng**:
- Thay đổi settings (min_uc, min_sc, dataset_code)
- Muốn re-preprocess với cấu hình mới
- Dọn dẹp disk space

---

### 2. `inspect_pickle.py`
**Mục đích**: Xem cấu trúc và thống kê của dataset đã preprocessing

**Sử dụng**:
```bash
cd ..
python tools/inspect_pickle.py
```

**Output**:
- Số lượng users, items
- Cấu trúc train/val/test
- Sample data
- User/Item mappings

**Khi nào dùng**:
- Sau khi chạy data_prepare.py
- Kiểm tra xem preprocessing có đúng không
- Debug dataset issues

---

### 3. `test_filtering.py`
**Mục đích**: Kiểm tra kết quả filtering (text/image)

**Sử dụng**:
```bash
cd ..
python tools/test_filtering.py
python tools/test_filtering.py --use_text
python tools/test_filtering.py --use_text --use_image
```

**Output**:
- Items với text/image/cả hai
- Phân tích metadata structure
- Sample items

**Khi nào dùng**:
- So sánh kết quả với/không filtering
- Hiểu tác động của use_text/use_image
- Debug filtering logic

---

### 4. `test_download_images.py`
**Mục đích**: Kiểm tra images đã download

**Sử dụng**:
```bash
cd ..
python tools/test_download_images.py
python tools/test_download_images.py --min_uc 20 --min_sc 20
```

**Output**:
- Số lượng images downloaded
- Tổng size (MB)
- Đường dẫn images folder
- Sample images với paths
- Verify files tồn tại

**Khi nào dùng**:
- Sau khi chạy với --use_image
- Kiểm tra images download thành công
- Debug image paths
- Xem dung lượng disk

---

## 🚀 Workflow thông thường:

### 1. Preprocessing lần đầu:
```bash
python data_prepare.py --use_text --use_image
```

### 2. Kiểm tra kết quả:
```bash
python tools/inspect_pickle.py
python tools/test_download_images.py
```

### 3. Nếu muốn thay đổi settings:
```bash
python tools/clean_preprocessed.py
python data_prepare.py --min_uc 10 --min_sc 10 --use_text --use_image
```

### 4. So sánh filtering:
```bash
python tools/test_filtering.py
python tools/test_filtering.py --use_text
```

---

## 📝 Lưu ý:

- Tất cả tools đều chạy từ **root folder** (inprocessing/)
- Dùng `cd ..` nếu đang ở trong folder tools/
- Tools không modify data, chỉ đọc và hiển thị
- Trừ `clean_preprocessed.py` sẽ **XÓA** data

---

## 🔧 Thêm tool mới:

Khi thêm utility script mới:
1. Đặt file vào folder `tools/`
2. Cập nhật README.md này
3. Đảm bảo script có thể chạy từ root folder
