import os, sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import torch, torchvision

frame_format, pixel_bytes = 'RGBA', 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp32').eval().to(device)
ssd_utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
detection_threshold = 0.4

Gst.init()
pipeline = Gst.parse_launch(f'''
    filesrc location=media/in.mp4 num-buffers=200 !
    decodebin !
    nvvideoconvert !
    video/x-raw,format={frame_format} !
    fakesink name=s
''')


def on_frame_probe(pad, info):
    buf = info.get_buffer()
    print(f'[{buf.pts / Gst.SECOND:6.2f}]')

    image_tensor = buffer_to_image_tensor(buf, pad.get_current_caps())
    image_batch = preprocess(image_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        detections_batch = detector(image_batch)
        postprocess(detections_batch)

    return Gst.PadProbeReturn.OK


def buffer_to_image_tensor(buf, caps):
    caps_structure = caps.get_structure(0)
    height, width = caps_structure.get_value('height'), caps_structure.get_value('width')

    is_mapped, map_info = buf.map(Gst.MapFlags.READ)
    if is_mapped:
        try:
            image_array = np.ndarray(
                (height, width, pixel_bytes),
                dtype=np.uint8,
                buffer=map_info.data
            )
            return torch.from_numpy(
                image_array[:,:,:3].copy() # RGBA -> RGB, and extend lifetime beyond subsequent unmap
            )
        finally:
            buf.unmap(map_info)


def preprocess(image_tensor):
    'take a centre crop 300x300'
    image_height, image_width, image_depth = image_tensor.size()
    copy_x, copy_y = min(300, image_width), min(300, image_height)

    dest_x_offset = max(0, (300 - image_width) // 2)
    source_x_offset = max(0, (image_width - 300) // 2)
    dest_y_offset = max(0, (300 - image_height) // 2)
    source_y_offset = max(0, (image_height - 300) // 2)

    input_tensor = torch.zeros((300, 300, 3), dtype=torch.float32, device=device)
    input_tensor[dest_y_offset:dest_y_offset + copy_y, dest_x_offset:dest_x_offset + copy_x] = \
        image_tensor[source_y_offset:source_y_offset + copy_y, source_x_offset:source_x_offset + copy_x]

    return torch.einsum(
        'hwc -> chw',
        normalize(input_tensor / 255)
    )


def normalize(input_tensor):
    'Nvidia SSD300 code uses mean and std-dev of 128/256'
    return (2.0 * input_tensor) - 1.0


def postprocess(detections_batch):
    results_batch = ssd_utils.decode_results(detections_batch)
    results_batch = [ssd_utils.pick_best(results, detection_threshold) for results in results_batch]
    for bboxes, classes, scores in results_batch:
        if scores.shape[0] > 0:
            print(bboxes, classes, scores)


pipeline.get_by_name('s').get_static_pad('sink').add_probe(
    Gst.PadProbeType.BUFFER,
    on_frame_probe
)

pipeline.set_state(Gst.State.PLAYING)

try:
    while True:
        msg = pipeline.get_bus().timed_pop_filtered(
            Gst.SECOND,
            Gst.MessageType.EOS | Gst.MessageType.ERROR
        )
        if msg:
            text = msg.get_structure().to_string() if msg.get_structure() else ''
            msg_type = Gst.message_type_get_name(msg.type)
            print(f'{msg.src.name}: [{msg_type}] {text}')
            break
finally:
    open(f'logs/{os.path.splitext(sys.argv[0])[0]}.pipeline.dot', 'w').write(
        Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL)
    )
    pipeline.set_state(Gst.State.NULL)
