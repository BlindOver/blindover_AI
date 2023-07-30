import os
import random
import argparse
import numpy as np
from PIL import Image, ImageEnhance
from glob import glob
from tqdm.auto import tqdm


def image_transformation(image: Image):
    methods = {
        'colr': np.arange(0.8, 1.3, 0.1), # 0.8 ~ 1.2
        'sharpness': np.arange(0.1, 5.1, 0.1), # 0.1 ~ 5.0
        'contrast': np.arange(0.7, 1.4, 0.1), # 0.7 ~ 1.3
        'brightness': np.arange(0.8, 1.3, 0.1), # 0.8 ~ 1.2
    }
    
    random_selected = random.choice(list(methods.keys()))
    factor = random.choice(methods[random_selected])
        
    if random_selected == 'color':
        changed_img = ImageEnhance.Color(image).enhance(factor)
    
    elif random_selected == 'sharpness':
        changed_img = ImageEnhance.Sharpness(image).enhance(factor)
        
    elif random_selected == 'contrast':
        changed_img = ImageEnhance.Contrast(image).enhance(factor)
        
    else:
        changed_img = ImageEnhance.Brightness(image).enhance(factor)
    
    return changed_img


def get_args_parser():
    parser = argparse.ArgumentParser(description='Image Composite', add_help=False)
    parser.add_argument('--foreground_path', type=str, required=True,
                        help='a path of foreground image or folder')
    parser.add_argument('--background_path', type=str, required=True,
                        help='a path of background image of folder')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='create a directory to save composite images')
    parser.add_argument('--limit', type=int, default=10,
                        help='limit the number of composited image for each class')
    
    return parser


# create new composited images of 10 for each class
def main(args):
    # create new folder
    new_folder = args.save_dir + 'composited'
    os.makedirs(new_folder, exist_ok=True)

    classes = [folder.split('/')[-1] for folder in glob(args.foreground_path+'**')]

    # try image composite for each class
    for class_ in classes:
        # creat a new folder for each class
        new_cls_folder = new_folder + '/' + class_
        os.makedirs(new_cls_folder, exist_ok=True)


        # load image files of background and foreground
        background_files = glob(args.background_path + '*.png')
        foreground_files = glob(args.foreground_path + class_ + '/*.png')
        
        # random selection
        background_files = random.sample(background_files, args.limit)
        foreground_files = random.sample(foreground_files, args.limit)

        # random shuffle
        random.shuffle(background_files)
        random.shuffle(foreground_files)

        # image composite
        for idx in tqdm(range(args.limit)):
            back = Image.open(background_files[idx]).convert('RGB')
            fore = Image.open(foreground_files[idx]).convert('RGB')

            back_w, back_h = back.size
            top_left = (int(back_w / 4), 0)
            bottom_right = (int(back_w / 4 * 3), back_h)
            back = back.crop(top_left + bottom_right)

            ratio = fore.size[1] / fore.size[0]
            height = back.size[1] * 0.8
            fore_h, fore_w = int(height), int(height / ratio)
            fore = fore.resize((fore_w, fore_h))

            back.paste(fore, (0, back.size[1] - fore.size[1]), fore) # composite
            back = image_transformation(back) # image transformation
            back.save(f'{new_cls_folder}/composited_{idx}.JPG') # saving image
            print(f'saved!   {new_cls_folder}/composited_{idx}.JPG')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Composite', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)