import os, sys
import math, time
import contextlib
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import torch, torchvision

frame_format, pixel_bytes, model_precision = 'RGBA', 4, 'fp32'
model_dtype = torch.float16 if model_precision == 'fp16' else torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=model_precision).eval().to(device)
ssd_utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
detection_threshold = 0.4
start_time, frames_processed = None, 0

# context manager to help keep track of ranges of time, using NVTX
@contextlib.contextmanager
def nvtx_range(msg):
    depth = torch.cuda.nvtx.range_push(msg)
    try:
        yield depth
    finally:
        torch.cuda.nvtx.range_pop()


def on_frame_probe(pad, info):
    global start_time, frames_processed
    start_time = start_time or time.time()

    with nvtx_range('on_frame_probe'):
        buf = info.get_buffer()
        print(f'[{buf.pts / Gst.SECOND:6.2f}]')

        image_tensor = buffer_to_image_tensor(buf, pad.get_current_caps())
        image_batch = preprocess(image_tensor.unsqueeze(0))
        frames_processed += image_batch.size(0)

        with torch.no_grad():
            with nvtx_range('inference'):
                locs, labels = detector(image_batch)
            postprocess(locs, labels)

        return Gst.PadProbeReturn.OK


def buffer_to_image_tensor(buf, caps):
    with nvtx_range('buffer_to_image_tensor'):
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


def preprocess(image_batch):
    '300x300 centre crop, normalize, HWC -> CHW'
    with nvtx_range('preprocess'):
        batch_dim, image_height, image_width, image_depth = image_batch.size()
        copy_x, copy_y = min(300, image_width), min(300, image_height)

        dest_x_offset = max(0, (300 - image_width) // 2)
        source_x_offset = max(0, (image_width - 300) // 2)
        dest_y_offset = max(0, (300 - image_height) // 2)
        source_y_offset = max(0, (image_height - 300) // 2)

        input_batch = torch.zeros((batch_dim, 300, 300, 3), dtype=model_dtype, device=device)
        input_batch[:, dest_y_offset:dest_y_offset + copy_y, dest_x_offset:dest_x_offset + copy_x] = \
            image_batch[:, source_y_offset:source_y_offset + copy_y, source_x_offset:source_x_offset + copy_x]

        return torch.einsum(
            'bhwc -> bchw',
            normalize(input_batch / 255)
        ).contiguous()


def normalize(input_tensor):
    'Nvidia SSD300 code uses mean and std-dev of 128/256'
    return (2.0 * input_tensor) - 1.0


def postprocess(locs, labels):
    with nvtx_range('postprocess'):
        results_batch = ssd_utils.decode_results((locs, labels))
        results_batch = [ssd_utils.pick_best(results, detection_threshold) for results in results_batch]
        for bboxes, classes, scores in results_batch:
            if scores.shape[0] > 0:
                print(bboxes, classes, scores)


Gst.init()
pipeline = Gst.parse_launch(f'''
    filesrc location=media/in.mp4 num-buffers=256 !
    decodebin !
    nvvideoconvert !
    video/x-raw,format={frame_format} !
    fakesink name=s
''')

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
    finish_time = time.time()
    open(f'logs/{os.path.splitext(sys.argv[0])[0]}.pipeline.dot', 'w').write(
        Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL)
    )
    pipeline.set_state(Gst.State.NULL)
    print(f'FPS: {frames_processed / (finish_time - start_time):.2f}')
