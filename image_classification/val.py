import os
import shutil

def organize_val_folder(path):
    val_dir = os.path.join(path, 'val')
    val_img_dir = os.path.join(val_dir, 'images')
    val_annotations = os.path.join(val_dir, 'val_annotations.txt')

    with open(val_annotations, 'r') as f:
        data = f.readlines()

    val_img_dict = {}
    for line in data:
        words = line.strip().split('\t')
        val_img_dict[words[0]] = words[1]

    # Cria subpastas por classe e move as imagens
    for img, folder in val_img_dict.items():
        new_path = os.path.join(val_dir, folder)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        shutil.move(os.path.join(val_img_dir, img), os.path.join(new_path, img))

    # Remove a pasta original de imagens (agora vazia)
    shutil.rmtree(val_img_dir)

# Exemplo de uso:
organize_val_folder("tiny-imagenet-200")