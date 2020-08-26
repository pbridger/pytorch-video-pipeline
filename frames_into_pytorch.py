
import os, sys

import numpy as np
import torch, torchvision

import gi
gi.require_version('Gst', '1.0')

from gi.repository import Gst

Gst.init()

frame_format, pixel_bytes = 'RGBA', 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

threshold_score = 0.5
person_label_index = 1
detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)

pipeline = Gst.parse_launch(f'''
    filesrc location=media/in.mp4 !
    decodebin !
    nvvideoconvert !
    video/x-raw,format={frame_format} !
    fakesink name=s
''')


def on_frame_probe(pad, info):
    buf = info.get_buffer()
    print(f'[{buf.pts / Gst.SECOND:6.2f}]')
    image_tensor = buffer_to_image_tensor(buf, pad.get_current_caps())[:3,:,:]

    with torch.no_grad():
        detections = detector(image_tensor.unsqueeze(0).to(device))[0]

    for bbox, label_index, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if score > threshold_score and label_index == person_label_index:
            print(f'[{buf.pts / Gst.SECOND:6.2f}] Person({score:4.1f}): {bbox}')

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
            ).copy() # extend array lifetime beyond subsequent unmap
            image_tensor = torch.einsum(
                'hwc -> chw',
                torch.from_numpy(image_array).float()
            )
            return image_tensor / 255 # 0..255 -> 0.0..1.0
        finally:
            buf.unmap(map_info)


pipeline.get_by_name('s').get_static_pad('sink').add_probe(
    Gst.PadProbeType.BUFFER,
    on_frame_probe
)


pipeline.set_state(Gst.State.PLAYING)

try:
    while True:
        msg = pipeline.get_bus().timed_pop_filtered(Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        if msg is None: continue

        structure = msg.get_structure()
        print(f'Message from {msg.src.name} [{Gst.msg(msg.type)}] {structure.to_string() if structure else ""}')

        if msg.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR):
            break
finally:
    Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, 'DONE')
    pipeline.set_state(Gst.State.NULL)

