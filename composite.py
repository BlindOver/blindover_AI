import os
import argparse
from PIL import Image
from glob import glob
from tqdm.auto import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser(description='Image Composite', add_help=False)
    parser.add_argument('--foreground_path', type=str, required=True,
                        help='a path of foreground image or folder')
    parser.add_argument('--background_path', type=str, required=True,
                        help='a path of background image of folder')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='create a directory to save composite images')
    return parser


def main(args):
    # create new folder
    new_folder = args.save_dir + 'composite'
    os.makedirs(new_folder, exist_ok=True)

    # load image files of background and foreground
    background_files = glob(args.background_path+'*.png')
    foreground_files = glob(args.foreground_path+'*.png')

    # image composite
    for idx, back_file in tqdm(enumerate(background_files)):
        for fore_file in foreground_files:
            back = Image.open(back_file)
            fore = Image.open(fore_file)

            back_w, back_h = back.size
            top_left = (int(back_w / 4), 0)
            bottom_right = (int(back_w / 4 * 3), back_h)
            back = back.drop(top_left + bottom_right)
            
            ratio = fore.size[1] / fore.size[0]
            height = back.size[1] * 0.8
            fore_h, fore_w = int(height), int(height / ratio)
            fore = fore.resize((fore_w, fore_h))

            back.paste(fore, (0, back.size[1] - fore.size[1]), fore)
            back.save(f'{args.save_dir}/composite_{idx}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Composite', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)