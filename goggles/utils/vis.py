import os

from PIL import ImageDraw
from torchvision import transforms


def get_image_from_tensor(x):
    transform_unnormalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1./0.229, 1./0.224, 1./0.225])
    transform_to_image = transforms.ToPILImage()

    ux = x.clone()
    if len(ux.size()) == 4:
        ux = ux[0]

    ux = transform_unnormalize(ux)
    image = transform_to_image(ux)
    return image


def get_image_with_patch_outline(
        x, corner_idx, image_patch_size, outline_color='black'):

    image = get_image_from_tensor(x)

    (i_nw, j_nw), (patch_w, patch_h) = corner_idx, image_patch_size

    x_nw, y_nw = j_nw, i_nw
    x_se, y_se = x_nw + patch_w, y_nw + patch_h

    draw = ImageDraw.Draw(image)
    draw.rectangle(((x_nw, y_nw), (x_se, y_se)), outline=outline_color)

    return image


def save_prototype_patch_visualization(model, dataset, nearest_patches_for_prototypes, outdir):
    for prototype_label, ((image_idx, patch_idx), nearest_patch) in nearest_patches_for_prototypes.items():
        attribute_name = dataset.get_attribute(prototype_label).name

        x = dataset[image_idx][0]

        image = get_image_with_patch_outline(x, *model.get_receptive_field(patch_idx))
        image.save(os.path.join(outdir, '%s.png' % attribute_name.replace('::', '-')))
