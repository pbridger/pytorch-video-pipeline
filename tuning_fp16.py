import os, sys
import math, time
import itertools
import contextlib
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import torch, torchvision

frame_format, pixel_bytes, model_precision = 'RGBA', 4, 'fp16'
model_dtype = torch.float16 if model_precision == 'fp16' else torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=model_precision).eval().to(device)
ssd_utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
detection_threshold = 0.4
start_time, frames_processed = None, 0
image_batch, batch_size = [], 8

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

    global image_batch

    if not image_batch:
        torch.cuda.nvtx.range_push('batch')
        torch.cuda.nvtx.range_push('create_batch')

    buf = info.get_buffer()
    print(f'[{buf.pts / Gst.SECOND:6.2f}]')

    image_tensor = buffer_to_image_tensor(buf, pad.get_current_caps())
    image_batch.append(image_tensor)

    if len(image_batch) < batch_size:
        return Gst.PadProbeReturn.OK

    torch.cuda.nvtx.range_pop() # create_batch

    image_batch = preprocess(torch.stack(image_batch))
    frames_processed += image_batch.size(0)

    with torch.no_grad():
        with nvtx_range('inference'):
            locs, labels = detector(image_batch)
            image_batch = []
        postprocess(locs, labels)

    torch.cuda.nvtx.range_pop() # batch
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


def init_dboxes():
    'adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/utils.py'
    fig_size = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    fk = fig_size / np.array(steps)

    dboxes = []
    # size of feature and number of feature
    for idx, sfeat in enumerate(feat_size):
        sk1 = scales[idx] / fig_size
        sk2 = scales[idx + 1] / fig_size
        sk3 = math.sqrt(sk1 * sk2)
        all_sizes = [(sk1, sk1), (sk3, sk3)]

        for alpha in aspect_ratios[idx]:
            w, h = sk1 * math.sqrt(alpha), sk1 / math.sqrt(alpha)
            all_sizes.append((w, h))
            all_sizes.append((h, w))

        for w, h in all_sizes:
            for i, j in itertools.product(range(sfeat), repeat=2):
                cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                dboxes.append((cx, cy, w, h))

    return torch.tensor(
        dboxes,
        dtype=model_dtype,
        device=device
    ).clamp(0, 1)


dboxes_xywh = init_dboxes().unsqueeze(dim=0)
scale_xy = 0.1
scale_wh = 0.2


def xywh_to_xyxy(bboxes_batch, scores_batch):
    bboxes_batch = bboxes_batch.permute(0, 2, 1)
    scores_batch = scores_batch.permute(0, 2, 1)

    bboxes_batch[:, :, :2] = scale_xy * bboxes_batch[:, :, :2]
    bboxes_batch[:, :, 2:] = scale_wh * bboxes_batch[:, :, 2:]

    bboxes_batch[:, :, :2] = bboxes_batch[:, :, :2] * dboxes_xywh[:, :, 2:] + dboxes_xywh[:, :, :2]
    bboxes_batch[:, :, 2:] = bboxes_batch[:, :, 2:].exp() * dboxes_xywh[:, :, 2:]

    # transform format to ltrb
    l, t, r, b = bboxes_batch[:, :, 0] - 0.5 * bboxes_batch[:, :, 2],\
                 bboxes_batch[:, :, 1] - 0.5 * bboxes_batch[:, :, 3],\
                 bboxes_batch[:, :, 0] + 0.5 * bboxes_batch[:, :, 2],\
                 bboxes_batch[:, :, 1] + 0.5 * bboxes_batch[:, :, 3]

    bboxes_batch[:, :, 0] = l
    bboxes_batch[:, :, 1] = t
    bboxes_batch[:, :, 2] = r
    bboxes_batch[:, :, 3] = b

    return bboxes_batch, torch.nn.functional.softmax(scores_batch, dim=-1)


def postprocess(locs, labels):
    with nvtx_range('postprocess'):
        locs, probs = xywh_to_xyxy(locs, labels)

        # flatten batch and classes
        batch_dim, box_dim, class_dim = probs.size()
        flat_locs = locs.reshape(-1, 4).repeat_interleave(class_dim, dim=0)
        flat_probs = probs.view(-1)
        class_indexes = torch.arange(class_dim, device=device).repeat(batch_dim * box_dim)
        image_indexes = (torch.ones(box_dim * class_dim, device=device) * torch.arange(1, batch_dim + 1, device=device).unsqueeze(-1)).view(-1)

        # only do NMS on detections over threshold, and ignore background (0)
        threshold_mask = (flat_probs > detection_threshold) & (class_indexes > 0)
        flat_locs = flat_locs[threshold_mask]
        flat_probs = flat_probs[threshold_mask]
        class_indexes = class_indexes[threshold_mask]
        image_indexes = image_indexes[threshold_mask]

        nms_mask = torchvision.ops.boxes.batched_nms(
            flat_locs,
            flat_probs,
            class_indexes * image_indexes,
            iou_threshold=0.7
        )

        bboxes = flat_locs[nms_mask].cpu()
        probs = flat_probs[nms_mask].cpu()
        class_indexes = class_indexes[nms_mask].cpu()
        if bboxes.size(0) > 0:
            print(bboxes, class_indexes, probs)


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
