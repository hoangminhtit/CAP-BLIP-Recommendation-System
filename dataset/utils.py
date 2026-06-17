import numpy as np
import pandas as pd
from tqdm import tqdm
import urllib.request
from urllib.error import URLError, HTTPError
import socket
import os
import hashlib
from PIL import Image
from io import BytesIO

from pathlib import Path
import zipfile
import tarfile
import sys


def download(url, savepath):
    urllib.request.urlretrieve(url, str(savepath))
    print()


def check_image_url(url, timeout=3):
    """
    Kiểm tra xem image URL có tồn tại và download được không.
    
    Args:
        url: URL của image
        timeout: Timeout cho request (giây)
    
    Returns:
        True nếu image tồn tại và download được, False nếu không
    """
    if not url or len(url.strip()) == 0:
        return False
    
    try:
        # Tạo request với timeout
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        
        # Chỉ lấy header, không download toàn bộ image
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # Kiểm tra status code
            if response.status != 200:
                return False
            
            # Kiểm tra content type có phải image không
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                return False
            
            return True
            
    except (URLError, HTTPError, socket.timeout, Exception):
        return False


def check_image_urls_batch(items_dict, max_workers=10):
    """
    Kiểm tra nhiều image URLs song song để tăng tốc độ.
    
    Args:
        items_dict: Dictionary {item_id: meta_info}
        max_workers: Số lượng threads song song
    
    Returns:
        Set các item_ids có image hợp lệ
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    valid_items = set()
    items_to_check = []
    
    # Chuẩn bị danh sách items cần kiểm tra
    for item_id, meta_info in items_dict.items():
        image_url = meta_info.get('image')
        if image_url and len(image_url.strip()) > 0:
            items_to_check.append((item_id, image_url))
    
    if not items_to_check:
        return valid_items
    
    # Kiểm tra song song
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(check_image_url, url): item_id 
            for item_id, url in items_to_check
        }
        
        for future in tqdm(as_completed(future_to_item), 
                          total=len(future_to_item), 
                          desc='Checking images'):
            item_id = future_to_item[future]
            try:
                if future.result():
                    valid_items.add(item_id)
            except Exception:
                pass  # Invalid image
    
    return valid_items


def download_image(url, save_path, timeout=10, retry=3):
    """
    Download image từ URL và lưu vào local.
    
    Args:
        url: URL của image
        save_path: Đường dẫn để lưu image
        timeout: Timeout cho request (giây)
        retry: Số lần thử lại nếu failed
    
    Returns:
        True nếu download thành công, False nếu thất bại
    """
    if not url or len(url.strip()) == 0:
        return False
    
    for attempt in range(retry):
        try:
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    continue
                
                # Đọc image data
                image_data = response.read()
                
                # Verify image có thể mở được bằng PIL
                try:
                    img = Image.open(BytesIO(image_data))
                    img.verify()  # Verify image integrity
                    
                    # Save image
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, 'wb') as f:
                        f.write(image_data)
                    
                    return True
                except Exception:
                    continue  # Invalid image format
                    
        except (URLError, HTTPError, socket.timeout, Exception):
            if attempt < retry - 1:
                continue
            else:
                return False
    
    return False


def get_image_filename(item_id, url):
    """
    Tạo tên file cho image dựa trên item_id và URL.
    
    Args:
        item_id: ID của item
        url: URL của image
    
    Returns:
        Tên file (ví dụ: "item_123_abc123.jpg")
    """
    # Lấy extension từ URL
    ext = url.split('.')[-1].split('?')[0].lower()
    if ext not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
        ext = 'jpg'  # Default extension
    
    # Tạo hash ngắn từ URL để tránh trùng lặp
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    
    return f"item_{item_id}_{url_hash}.{ext}"


def download_and_verify_images_batch(items_dict, image_folder, max_workers=10):
    """
    Download và verify nhiều images song song (VỪA CHECK VỪA DOWNLOAD - chỉ 1 lần request).
    Thay thế cho việc check_image_urls_batch() rồi mới download_images_batch().
    
    Args:
        items_dict: Dictionary {item_id: meta_info} với meta_info có key 'image'
        image_folder: Folder để lưu images
        max_workers: Số lượng threads song song
    
    Returns:
        Tuple (downloaded_images, valid_items)
        - downloaded_images: Dictionary {item_id: local_image_path} cho các images download thành công
        - valid_items: Set các item_ids có image hợp lệ
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    image_folder = Path(image_folder)
    image_folder.mkdir(parents=True, exist_ok=True)
    
    downloaded_images = {}
    valid_items = set()
    items_to_download = []
    
    # Chuẩn bị danh sách items cần download
    for item_id, meta_info in items_dict.items():
        image_url = meta_info.get('image')
        if image_url and len(image_url.strip()) > 0:
            filename = get_image_filename(item_id, image_url)
            save_path = image_folder / filename
            items_to_download.append((item_id, image_url, str(save_path)))
    
    if not items_to_download:
        return downloaded_images, valid_items
    
    print(f'Downloading and verifying {len(items_to_download)} images (1 request per image)...')
    
    # Download song song - VỪA CHECK VỪA DOWNLOAD
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(download_image, url, save_path): (item_id, save_path)
            for item_id, url, save_path in items_to_download
        }
        
        for future in tqdm(as_completed(future_to_item), 
                          total=len(future_to_item), 
                          desc='Downloading & verifying images'):
            item_id, save_path = future_to_item[future]
            try:
                if future.result():
                    downloaded_images[item_id] = save_path
                    valid_items.add(item_id)
            except Exception:
                pass  # Failed to download
    
    print(f'Successfully downloaded {len(downloaded_images)}/{len(items_to_download)} images')
    
    return downloaded_images, valid_items


def download_images_batch(items_dict, image_folder, max_workers=10):
    """
    DEPRECATED: Sử dụng download_and_verify_images_batch() thay thế để tối ưu hơn.
    
    Download nhiều images song song.
    """
    downloaded_images, _ = download_and_verify_images_batch(items_dict, image_folder, max_workers)
    return downloaded_images


def unzip(zippath, savepath):
    print("Extracting data...")
    zip = zipfile.ZipFile(zippath)
    zip.extractall(savepath)
    zip.close()


def unziptargz(zippath, savepath):
    print("Extracting data...")
    f = tarfile.open(zippath)
    f.extractall(savepath)
    f.close()


def normalize_text(text: str) -> str:
    """
    Normalize text according to spec:
    - lowercase
    - remove special characters (keep only alphanumeric and spaces)
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters (keep only alphanumeric and spaces)
    import re
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Collapse multiple spaces into single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def truncate_text(text: str, max_length: int, from_end: bool = True) -> str:
    """
    Truncate text to max_length characters.
    
    Args:
        text: Input text string
        max_length: Maximum length in characters
        from_end: If True, truncate from the end (keep beginning). If False, truncate from beginning.
        
    Returns:
        Truncated text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    if len(text) <= max_length:
        return text
    
    if from_end:
        # Truncate from the end (keep beginning)
        return text[:max_length - 3] + "..."
    else:
        # Truncate from the beginning (keep end)
        return "..." + text[-(max_length - 3):]


def process_item_text(title: str, description: str, max_length: int = 512) -> str:
    """
    Process item text according to spec:
    1. Concatenate title + description
    2. Normalize (lowercase, remove special chars)
    3. Truncate to max_length (from end)
    
    Args:
        title: Item title
        description: Item description
        max_length: Maximum text length (default: 512, configurable 256-512)
        
    Returns:
        Processed text string
    """
    # Step 1: Concatenate title + description
    title = str(title).strip() if title else ""
    description = str(description).strip() if description else ""
    text = f"{title} {description}".strip()
    
    if not text:
        return ""
    # Step 2: Normalize (lowercase, remove special chars)
    text = normalize_text(text)
    
    # Step 3: Truncate from end
    text = truncate_text(text, max_length, from_end=True)
    
    return text