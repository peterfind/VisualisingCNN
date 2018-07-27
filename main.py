import os
import cv2
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

def load_checkpoint():
    """
        Loads the checkpoint of the trained model and returns the model.
    """
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        checkpoint = torch.load(opt.model)
    else:
        checkpoint = torch.load(
            opt.model, map_location=lambda storage, loc: storage)

    pretrained_model = models.resnet50(pretrained=True)
    num_ftrs = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_ftrs, 2)

    if use_gpu:
        pretrained_model = pretrained_model.cuda()

    pretrained_model.load_state_dict(checkpoint)
    pretrained_model.eval()

    return pretrained_model


def preprocess_image(cv2im, resize_im=True):
    """
        Resizing the image as per parameter, converts it to a torch tensor and returns
        torch variable.
    """
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def forward_pass_on_convolutions(self, x):
    """
        Does a forward pass on convolutions, hooks the function at given layer
    """
    conv_output = None
    for module_name, module in self.model._modules.items():
        print(module_name)
        if module_name == 'fc':
            return conv_output, x
        x = module(x)  # Forward
        if module_name == self.target_layer:
            print('True')
            x.register_hook(self.save_gradient)
            conv_output = x  # Save the convolution output on that layer
    return conv_output, x

def forward_pass(self, x):
    """
        Does a full forward pass on the model
    """
    # Forward pass on the convolutions
    conv_output, x = self.forward_pass_on_convolutions(x)
    x = x.view(x.size(0), -1)  # Flatten
    # Forward pass on the classifier
    x = self.model.fc(x)
    return conv_output, x

def generate_cam(self, input_image, target_index=None):
    """
        Full forward pass
        conv_output is the output of convolutions at specified layer
        model_output is the final output of the model
    """
    conv_output, model_output = self.extractor.forward_pass(input_image)
    if target_index is None:
        target_index = np.argmax(model_output.data.numpy())
    # Target for backprop
    one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
    one_hot_output[0][target_index] = 1
    # Zero grads
    self.model.fc.zero_grad()
    # Backward pass with specified target
    model_output.backward(gradient=one_hot_output, retain_graph=True)
    # Get hooked gradients
    guided_gradients = self.extractor.gradients.data.numpy()[0]
    # Get convolution outputs
    target = conv_output.data.numpy()[0]
    # Get weights from gradients
    # Take averages for each gradient
    weights = np.mean(guided_gradients, axis=(1, 2))
    # Create empty numpy array for cam
    cam = np.ones(target.shape[1:], dtype=np.float32)
    # Multiply each weight with its conv output and then, sum
    for i, w in enumerate(weights):
        cam += w * target[i, :, :]
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) -
                                 np.min(cam))  # Normalize between 0-1
    cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
    return cam


def save_class_activation_on_image(org_img, activation_map, file_name):
    """
      Saves the activation map as a heatmap imposed on the original image.
    """
    if not os.path.exists('./results'):
        os.makedirs('./results')
    # Grayscale activation map
    path_to_file = os.path.join('./results', file_name + '_Cam_Grayscale.jpg')
    cv2.imwrite(path_to_file, activation_map)
    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    path_to_file = os.path.join('./results', file_name + '_Cam_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)
    # Heatmap on picture
    org_img = cv2.resize(org_img, (224, 224))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join('./results', file_name + '_Cam_On_Image.jpg')
    cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))

''''''