import os
import zipfile

def zip_project(output_filename):
    # Các thư mục/file muốn loại bỏ khỏi file zip
    exclude_dirs = {'archive', 'artifacts', '.venv', '.git', '.agent', '__pycache__', '.ipynb_checkpoints'}
    exclude_files = {output_filename, 'Flickr8k_Dataset.zip', 'Flickr8k_text.zip', 'download_data.py', 'create_zip.py'}

    print(f"Đang tạo file {output_filename}...")
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # Loại bỏ các thư mục không mong muốn
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            
            for file in files:
                if file in exclude_files or file.endswith('.pyc'):
                    continue
                
                file_path = os.path.join(root, file)
                # Tính toán path tương đối để file zip đẹp hơn
                arcname = os.path.relpath(file_path, '.')
                print(f"  Adding: {arcname}")
                zipf.write(file_path, arcname)

    print(f"Hoàn tất! File đã được tạo tại: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    zip_project("image_captioning_kaggle.zip")
