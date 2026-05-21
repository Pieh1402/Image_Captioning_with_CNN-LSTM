import os
import requests
import zipfile
import pandas as pd
from tqdm import tqdm

def download_file(url, dest):
    print(f"Đang tải {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(dest, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

def setup_data():
    base_dir = os.getcwd()
    archive_dir = os.path.join(base_dir, "archive")
    images_dir = os.path.join(archive_dir, "Images")
    
    os.makedirs(images_dir, exist_ok=True)
    
    img_zip = os.path.join(base_dir, "Flickr8k_Dataset.zip")
    txt_zip = os.path.join(base_dir, "Flickr8k_text.zip")
    
    # Links
    img_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    txt_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    
    # Download if not exists
    if not os.path.exists(img_zip):
        download_file(img_url, img_zip)
    if not os.path.exists(txt_zip):
        download_file(txt_url, txt_zip)
        
    # Extract images
    print("Đang giải nén ảnh...")
    with zipfile.ZipFile(img_zip, 'r') as zip_ref:
        zip_ref.extractall(archive_dir)
    
    # Extract text
    print("Đang giải nén captions...")
    with zipfile.ZipFile(txt_zip, 'r') as zip_ref:
        zip_ref.extractall(archive_dir)
        
    # Move images to archive/Images if they were extracted to a subfolder
    extracted_img_dir = os.path.join(archive_dir, "Flicker8k_Dataset")
    if os.path.exists(extracted_img_dir):
        print("Đang sắp xếp lại thư mục ảnh...")
        for img_name in os.listdir(extracted_img_dir):
            os.rename(os.path.join(extracted_img_dir, img_name), os.path.join(images_dir, img_name))
        os.rmdir(extracted_img_dir)
        
    # Convert token.txt to captions.txt (CSV)
    print("Đang chuyển đổi định dạng captions...")
    token_file = os.path.join(archive_dir, "Flickr8k.token.txt")
    output_csv = os.path.join(archive_dir, "captions.txt")
    
    data = []
    with open(token_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.split('\t')
            if len(parts) < 2: continue
            img_id = parts[0].split('#')[0]
            caption = parts[1].strip()
            data.append([img_id, caption])
            
    df = pd.DataFrame(data, columns=['image', 'caption'])
    df.to_csv(output_csv, index=False)
    print(f"Đã tạo {output_csv} với {len(df)} dòng.")
    
    # Cleanup
    print("Đang dọn dẹp...")
    # Optional: os.remove(img_zip)
    # Optional: os.remove(txt_zip)
    print("Hoàn tất!")

if __name__ == "__main__":
    setup_data()
