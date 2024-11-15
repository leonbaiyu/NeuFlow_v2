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

imPath = '/home/leonwilliams/workshop/pangu/panguData/flight08'
vis_path = os.path.join(imPath,os.path.basename(imPath)+'_results_neuflow_vert_rotated_v2')

if not os.path.isdir(vis_path):
    os.mkdir(vis_path)

image_path_list = sorted(glob(os.path.join(imPath,'*.png')))

def draw_optical_flow(image, u_flow, v_flow, grid_size=(6, 6), scale=10):
    h, w, _ = image.shape
    grid_h, grid_w = grid_size
    h_step, w_step = h/grid_h, w/grid_w
    arrow_x = np.linspace(0, w, grid_w, endpoint=False, dtype=int)
    arrow_y = np.linspace(0, h, grid_h, endpoint=False, dtype=int)

    output_image = image.copy()

    for y in arrow_y:
        for x in arrow_x:
            u = u_flow[y, x]
            v = v_flow[y, x]
            # magnitude = np.sqrt(u**2 + v**2)
            # angle = np.arctan2(v, u)
            color = (0, 255, 0)
            cv2.arrowedLine(output_image, (x+int(w_step/2), y+int(h_step/2)), (x+int(w_step/2 + scale * u), y+int(h_step/2 + scale * v)), color, thickness = 1, tipLength = 0.3)

    return output_image

def get_cuda_image(image_path):
    image = cv2.imread(image_path)

    # rotate the image
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

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


def rotate_vectors_90_degrees(vectors):
    """
    Rotates an N x M array of 2D vectors by 90 degrees clockwise.

    Parameters:
    vectors (numpy.ndarray): An array of shape (N, M, 2), where each vector is [x, y].

    Returns:
    numpy.ndarray: An array of the same shape with the vectors rotated by 90 degrees clockwise.
    """
    # Ensure the input is a NumPy array
    vectors = np.asarray(vectors)
    
    if vectors.shape[-1] != 2:
        raise ValueError("The input array must have the last dimension of size 2 (2D vectors).")
    
    # Rotate each vector by swapping components and negating the first component
    rotated = np.empty_like(vectors)
    rotated[..., 0] = vectors[..., 1]   # x' = y
    rotated[..., 1] = -vectors[..., 0]  # y' = -x
    
    return rotated

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

flow_tensor = []

for image_path_0, image_path_1 in zip(image_path_list[:-1], image_path_list[1:]):

    print(image_path_0)

    image_0 = get_cuda_image(image_path_0)
    image_1 = get_cuda_image(image_path_1)

    file_name = os.path.basename(image_path_0)

    with torch.no_grad():

        flow = model(image_0, image_1)[-1][0]

        flow = flow.permute(1,2,0).cpu().numpy()

        image_0 = cv2.imread(image_path_0)

        # rotate flow matrix 90 degrees counterclockwise
        flow = cv2.rotate(flow, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # rotate flow vectors 90 degrees
        flow = rotate_vectors_90_degrees(flow)

        # resize flow to have x and y dimensions corresponding to image_0
        flow = cv2.resize(flow.astype(np.float32), (image_0.shape[1], image_0.shape[0]))

        # Append flows to flow tensor
        flow_tensor.append(flow)

        # Extract u and v maps from OF NN output
        u_flow = flow[:, :, 0]
        v_flow = flow[:, :, 1]

        # storing the standard colour mapped dense optical flow representation
        # flow_coloured_image = flow_viz.flow_to_image(flow)

        # Creating image with arrows in vector map style of depiction
        arrowed_image = draw_optical_flow(image_0,u_flow,v_flow)

        # stacking frames vertically for comparison
        # cv2.imwrite(vis_path + file_name, np.vstack([flow_coloured_image,arrowed_image]))   

        # Alternatively just outputting the returned arrowed_image
        cv2.imwrite(os.path.join(vis_path,file_name), arrowed_image)

flow_tensor = np.array(flow_tensor)

np.save(os.path.join(vis_path, os.path.basename(vis_path)+"_flow_tensor.npy"),flow_tensor)