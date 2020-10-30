
from PIL import Image, ImageDraw, ImageFont
from flags import FLAGS
import os
import numpy as np
from utils.preprocessing import label_colours

# data path for predicted mask
pred_path = '../dataset/VOCdevkit/pred/'


# generate mask from label
def label2mask(label_array):
    """it take a 2D array as input
    and output an image object
    """
    img = Image.new('RGB', (label_array.shape[1], label_array.shape[0]))
    pixels = img.load()
    for j_, j in enumerate(label_array):
        for k_, k in enumerate(j):
            if k <= len(label_colours):
                pixels[k_, j_] = label_colours[k]
    return img

def main():
    # file name for image, label
    file_path = FLAGS.evaluation_data_list
    file_list =[filename.strip() for filename in open(file_path, 'r').readlines()]
    image_files = [os.path.join(FLAGS.image_dir, filename) + '.jpg' for filename in file_list]
    label_files = [os.path.join(FLAGS.eval_label_dir, filename) + '.png' for filename in file_list]

    eval_metrics = open(os.path.join(pred_path, 'eval_metric.txt')).readlines()
    eval_metrics = [line.strip().split(',') for line in eval_metrics]

    if not os.path.exists(os.path.join(pred_path, 'compare/')):
        os.mkdir(os.path.join(pred_path, 'compare/'))

    for i in range(len(image_files)):
        try:
            # original image
            im = Image.open(image_files[i])
            # ground truth label
            i_label = Image.open(label_files[i])
            label_array = np.asarray(i_label)
            im_label = label2mask(label_array)
            # predicted mask
            im_pred = Image.open(os.path.join(pred_path, 'pred_mask/mask' + str(i+1)+'.jpg' ))

            # combine the three images
            images = [im, im_label, im_pred]
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            new_im = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for im in images:
                new_im.paste(im, (x_offset, 0))
                x_offset += im.size[0]

            fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 24)
            d = ImageDraw.Draw(new_im)
            d.text((0, 0), eval_metrics[i][1] + eval_metrics[i][3],
                   font = fnt, fill=(153, 0, 27))

            # save the output to the following path
            print('step {} file {} was saved'.format(i, image_files[i].split('/')[-1]))
            new_im.save(os.path.join(pred_path, 'compare/' + image_files[i].split('/')[-1]))
        except:
            print('file {} was not able to process'.format(image_files[i].split('/')[-1]))

if __name__ == '__main__':
    main()