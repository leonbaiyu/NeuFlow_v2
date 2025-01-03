import torch
from glob import glob
import os
import numpy as np
import cv2
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock
from data_utils import flow_viz


image_width = 768
image_height = 432

def draw_optical_flow(image, u_flow, v_flow, grid_size=(20, 20), scale=2):
    h, w, _ = image.shape
    grid_h, grid_w = grid_size
    arrow_x = np.linspace(0, w, grid_w, endpoint=False, dtype=int)
    arrow_y = np.linspace(0, h, grid_h, endpoint=False, dtype=int)

    output_image = image.copy()

    for y in arrow_y:
        for x in arrow_x:
            u = u_flow[y, x]
            v = v_flow[y, x]
            magnitude = np.sqrt(u**2 + v**2)
            angle = np.arctan2(v, u)
            color = (0, 255, 0)
            cv2.arrowedLine(output_image, (x, y), (int(x + scale * u), int(y + scale * v)), color, 2)

    return output_image

def get_cuda_image(image_path):
    image = cv2.imread(image_path)

    image = cv2.resize(image, (image_width, image_height))

    image = torch.from_numpy(image).permute(2, 0, 1).half()
    return image[None].cuda()


def fuse_conv_and_bn(conv, bn):
        """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
        fusedconv = (
            torch.nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )

        # Prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # Prepare spatial bias
        b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv


image_path_list = sorted(glob('/home/leonwilliams/workshop/pangu/NeuFlow_v2/test_images/*.jpg'))
vis_path = '/home/leonwilliams/workshop/pangu/NeuFlow_v2/test_results/'

device = torch.device('cuda')

model = NeuFlow().to(device)

checkpoint = torch.load('/home/leonwilliams/workshop/pangu/NeuFlow_v2/neuflow_mixed.pth', map_location='cuda')

model.load_state_dict(checkpoint['model'], strict=True)

for m in model.modules():
    if type(m) is ConvBlock:
        m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
        m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
        delattr(m, "norm1")  # remove batchnorm
        delattr(m, "norm2")  # remove batchnorm
        m.forward = m.forward_fuse  # update forward

model.eval()
model.half()

model.init_bhwd(1, image_height, image_width, 'cuda')

if not os.path.exists(vis_path):
    os.makedirs(vis_path)

for image_path_0, image_path_1 in zip(image_path_list[:-1], image_path_list[1:]):

    print(image_path_0)

    image_0 = get_cuda_image(image_path_0)
    image_1 = get_cuda_image(image_path_1)

    file_name = os.path.basename(image_path_0)

    with torch.no_grad():

        flow = model(image_0, image_1)[-1][0]

        flow = flow.permute(1,2,0).cpu().numpy()

        # Extract u and v maps from OF NN output
        u_flow = flow[:, :, 0]
        v_flow = flow[:, :, 1]

        # Resizing original image to have same dimensions as image_0
        image_0 = cv2.resize(cv2.imread(image_path_0),(image_width, image_height))

        # storing the standard colour mapped dense optical flow representation
        flow_coloured_image = flow_viz.flow_to_image(flow)

        # Creating image with arrows in vector map style of depiction
        arrowed_image = draw_optical_flow(image_0,u_flow,v_flow, grid_size = (20,20),scale=2)

        # stacking frames vertically for comparison
        cv2.imwrite(vis_path + file_name, np.vstack([image_0,flow_coloured_image,arrowed_image]))   
