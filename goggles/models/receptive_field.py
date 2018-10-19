import torch.nn as nn

# Receptive Field is a utility class with functions related to the receptive field of the output in the input image
# Jargon : patch - patch can refer to a rectangle of input or the output image
#        : coordinates - (i_nw, j_nw) - top left coordinates, (i_se, j_se) - bottom right coordinates


class ReceptiveField(nn.Module):
    def __init__(self,  o_i, o_j, image_size, num_conv, kernel_size=2, conv_stride=2, conv_padding=0):

        # set params of the model
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.image_size = image_size

        ## Compute input_image_size during forward pass
        #input_image_size stores the size of input before every convolution
        input_image_size = [0 for i in range(num_conv)]
        input_image_size[0] = image_size
        for i in range(1, num_conv):
            input_image_size[i] = 1 + (input_image_size[i-1] - kernel_size + 2*conv_padding)//conv_stride

        # Compute receptive field by backtracking
        i_nw, j_nw, i_se, j_se = self.get_receptive_field( o_i, o_j, input_image_size[-1])

        for i in range(num_conv - 1, 0, -1):
            i_nw, j_nw, _, _ = self.get_receptive_field(i_nw, j_nw, input_image_size[i - 1])
            _, _, i_se, j_se = self.get_receptive_field(i_se, j_se, input_image_size[i - 1])

        self.receptive_field_nw = (i_nw, j_nw)
        self.receptive_field_se = (i_se, j_se)


    def get_receptive_field(self,  o_i, o_j, image_size):
        i_nw, j_nw = o_i * self.conv_stride - self.conv_padding, o_j * self.conv_stride - self.conv_padding

        # -1 is for inclusive end indices
        receptive_field_size_i = self.kernel_size + min(0, i_nw) + min(0, image_size - i_nw - self.kernel_size - 1)
        receptive_field_size_j = self.kernel_size + min(0, j_nw) + min(0, image_size - j_nw - self.kernel_size - 1)

        i_nw, j_nw = max(i_nw, 0), max(j_nw, 0)
        i_se, j_se = i_nw + receptive_field_size_i, j_nw + receptive_field_size_j

        return i_nw, j_nw, i_se, j_se


## Test Case
## image size 5 and 2 convolutions
## receptive fields for output kernel size 2 - (0,0), (3,3)   (0,1),(3,4)
##                                             (1,0), (4,3)   (1,1), (4,4)

if __name__ == '__main__':
    import torch

    input_image_size = 5

    rf = ReceptiveField(1, 1, input_image_size, 2, 2, 2, 1)
    print(rf.receptive_field_nw, rf.receptive_field_se)
