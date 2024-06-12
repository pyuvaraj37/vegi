import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
from collections import OrderedDict
import onnxruntime
import onnx

import cv2
import numpy as np
from .craft_utils import getDetBoxes, adjustResultCoordinates
from .imgproc import resize_aspect_ratio, normalizeMeanVariance
from .craft import CRAFT

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def test_net(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, poly, device, estimate_num_chars=False):
    if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
        image_arrs = image
    else:                                                        # image is single numpy array
        image_arrs = [image]

    img_resized_list = []
    # resize
    for img in image_arrs:
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, canvas_size,
                                                                      interpolation=cv2.INTER_LINEAR,
                                                                      mag_ratio=mag_ratio)
        img_resized_list.append(img_resized)
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing
    x = [np.transpose(normalizeMeanVariance(n_img), (2, 0, 1))
         for n_img in img_resized_list]
    x = torch.from_numpy(np.array(x))
    x = x.to(device)

    ### Onnx Exporting ###

    # batch_size_1 = 500
    # batch_size_2 = 500
    # in_shape=[1, 3, batch_size_1, batch_size_2]
    # dummy_input = torch.rand(in_shape)
    # dummy_input = dummy_input.to(device)

    # model_save_path = "./onnx_models/detector.onnx"
    # torch.onnx.export(
    #     net.module,
    #     dummy_input,
    #     model_save_path,
    #     export_params=True,
    #     opset_version=11,
    #     input_names = ['input'],
    #     output_names = ['output'],
    #     dynamic_axes={'input' : {2 : 'batch_size_1', 3: 'batch_size_2'}},
    # )

    # onnx_model = onnx.load("detectionModel.onnx")
    # try:
    #     onnx.checker.check_model(onnx_model)
    # except onnx.checker.ValidationError as e:
    #     print('The model is invalid: %s' % e)
    # else:
    #     print('The model is valid!')
    
                            ### #### ###

    # forward pass
    with torch.no_grad():
        y, feature = net(x) #Comment for onnx
        ######################### ONNX CHANGES ##########################
        onnx_model_path = "C://Users//mikuv//Desktop//ryzen-ai-sw-1.1//RyzenAI-SW//easy-onnx//models//detector.onnx"
        # onnx_model_path = "C://Users//mikuv//Desktop//ryzen-ai-sw-1.1//RyzenAI-SW//easy-onnx//models//detector_quantized.onnx"
        sess_options = onnxruntime.SessionOptions()
        config_file_path = "C://Users//mikuv//Desktop//ryzen-ai-sw-1.1//RyzenAI-SW//easy-onnx//vaip_config.json"

        ort_session = onnxruntime.InferenceSession(onnx_model_path, 
                                                #    providers = ['CPUExecutionProvider'],
                                                   providers = ['VitisAIExecutionProvider'],
                                                   sess_options=sess_options,
                                                   provider_options=[{'config_file': config_file_path}]
                                                   )
        
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)
        y = ort_outs[0]
        print("onnx detector ran")
    ##################################################################

    boxes_list, polys_list = [], []
    for out in y:
        # make score and link map
        # score_text = out[:, :, 0].cpu().data.numpy() #Comment for onnx
        # score_link = out[:, :, 1].cpu().data.numpy() #Comment for onnx
        ######################### ONNX CHANGES ##########################
        score_text = out[:, :, 0]
        score_link = out[:, :, 1]
        ##################################################################

        # Post-processing
        boxes, polys, mapper = getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly, estimate_num_chars)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        if estimate_num_chars:
            boxes = list(boxes)
            polys = list(polys)
        for k in range(len(polys)):
            if estimate_num_chars:
                boxes[k] = (boxes[k], mapper[k])
            if polys[k] is None:
                polys[k] = boxes[k]
        boxes_list.append(boxes)
        polys_list.append(polys)

    return boxes_list, polys_list

def get_detector(trained_model, device='cpu', quantize=True, cudnn_benchmark=False):
    net = CRAFT()

    if device == 'cpu':
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
        if quantize:
            try:
                torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = cudnn_benchmark

    net.eval()
    return net

def get_textbox(detector, image, canvas_size, mag_ratio, text_threshold, link_threshold, low_text, poly, device, optimal_num_chars=None, **kwargs):
    result = []
    estimate_num_chars = optimal_num_chars is not None
    bboxes_list, polys_list = test_net(canvas_size, mag_ratio, detector,
                                       image, text_threshold,
                                       link_threshold, low_text, poly,
                                       device, estimate_num_chars)
    if estimate_num_chars:
        polys_list = [[p for p, _ in sorted(polys, key=lambda x: abs(optimal_num_chars - x[1]))]
                      for polys in polys_list]

    for polys in polys_list:
        single_img_result = []
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            single_img_result.append(poly)
        result.append(single_img_result)

    return result
