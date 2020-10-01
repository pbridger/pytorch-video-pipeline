import os, sys
import math, time
import itertools
import contextlib
import copy
import threading, queue
import gil_load
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import torch, torchvision
import ghetto_nvds

frame_format, pixel_bytes, model_precision = 'RGBA', 4, 'fp16'
model_dtype = torch.float16 if model_precision == 'fp16' else torch.float32
detection_threshold = 0.4
start_time, frames_processed = time.time(), 0
batch_size, num_inference_threads = 8, 2
num_devices = torch.cuda.device_count()
detector = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=model_precision).eval()


# context manager to help keep track of ranges of time, using NVTX
@contextlib.contextmanager
def nvtx_range(msg):
    depth = torch.cuda.nvtx.range_push(msg)
    try:
        yield depth
    finally:
        torch.cuda.nvtx.range_pop()


create_tensor_stream = torch.cuda.Stream()

def on_frame_probe(pad, info):
    global start_time, frames_processed
    buf = info.get_buffer()
    # print(f'[{buf.pts / Gst.SECOND:6.2f}]')
    device, detector, dboxes, image_queue = thread_contexts[frames_processed % len(thread_contexts)]

    with torch.no_grad():
        with torch.cuda.stream(create_tensor_stream):
            image_tensor = buffer_to_image_tensor(device, buf, pad.get_current_caps())
            image_queue.put((image_tensor, torch.cuda.Event()))

            start_time = time.time() if frames_processed == 0 else start_time
            frames_processed += 1
            return Gst.PadProbeReturn.OK


def buffer_to_image_tensor(device, buf, caps):
    with nvtx_range('buffer_to_image_tensor'):
        caps_structure = caps.get_structure(0)
        height, width = caps_structure.get_value('height'), caps_structure.get_value('width')

        is_mapped, map_info = buf.map(Gst.MapFlags.READ)
        if is_mapped:
            try:
                source_surface = ghetto_nvds.NvBufSurface(map_info)
                torch_surface = ghetto_nvds.NvBufSurface(map_info)

                dest_tensor = torch.zeros(
                    (torch_surface.surfaceList[0].height, torch_surface.surfaceList[0].width, 4),
                    dtype=torch.uint8,
                    device=device
                )

                torch_surface.struct_copy_from(source_surface)
                assert(source_surface.numFilled == 1)
                assert(source_surface.surfaceList[0].colorFormat == 19) # RGBA

                # make torch_surface map to dest_tensor memory
                torch_surface.surfaceList[0].dataPtr = dest_tensor.data_ptr()
                torch_surface.gpuId = device.index

                # copy decoded GPU buffer (source_surface) into Pytorch tensor (torch_surface -> dest_tensor)
                torch_surface.mem_copy_from(source_surface)
            finally:
                buf.unmap(map_info)

            return dest_tensor[:, :, :3]


def inference_thread_f(device, detector, dboxes, image_queue):
    cuda_stream = torch.cuda.Stream(device)

    while True:
        images, events = [], []
        while len(images) < batch_size:
            next_image, image_event = image_queue.get()
            if next_image is None:
                return None
            images.append(next_image)
            events.append(image_event)

        with torch.cuda.stream(cuda_stream):
            with torch.no_grad():
                for e in events:
                    e.synchronize()

                image_batch = preprocess(device, torch.stack(images))

                with nvtx_range('inference'):
                    locs, labels = detector(image_batch)
                    image_batch = []
                postprocess(device, dboxes, locs, labels)


def preprocess(device, image_batch):
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


def init_dboxes(device):
    'adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/utils.py'
    fig_size = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    fk = fig_size / torch.tensor(steps).float()

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


scale_xy = 0.1
scale_wh = 0.2


def xywh_to_xyxy(dboxes_xywh, bboxes_batch, scores_batch):
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


def postprocess(device, dboxes, locs, labels):
    with nvtx_range('postprocess'):
        locs, probs = xywh_to_xyxy(dboxes, locs, labels)

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
        # if bboxes.size(0) > 0:
        #     print(bboxes, class_indexes, probs)


if num_devices:
    thread_contexts = []

    for device_idx in range(num_devices):
        device = torch.device(f'cuda:{device_idx}')
        device_detector = copy.deepcopy(detector).to(device)
        dboxes_xywh = init_dboxes(device).unsqueeze(dim=0)

        for inference_idx in range(num_inference_threads):
            thread_queue = queue.Queue(2 * batch_size)
            thread_contexts.append((device, device_detector, dboxes_xywh, thread_queue))

else:
    sys.exit(1)

try:
    gil_load.init()
    gil_load_enabled = True
except RuntimeError:
    gil_load_enabled = False

Gst.init()
pipeline = Gst.parse_launch(f'''
    filesrc location=media/in.mp4 num-buffers=2048 !
    decodebin !
    nvvideoconvert !
    video/x-raw(memory:NVMM),format={frame_format} !
    fakesink name=s
''')

pipeline.get_by_name('s').get_static_pad('sink').add_probe(
    Gst.PadProbeType.BUFFER,
    on_frame_probe
)

inference_threads = []
for device, detector, dboxes, image_queue in thread_contexts:
    inference_threads.append(
        threading.Thread(target=inference_thread_f, args=(device, detector, dboxes, image_queue))
    )
    inference_threads[-1].start()

# for each thread doing the pointless gil_10_pc, the GIL is busy an additional ~10% of time
def gil_10_pc():
    while True:
        for i in range(300):
            a = 1 + 1
        time.sleep(1e-9)

gil_threads = []
for gil_idx in range(0):
    gil_threads.append(threading.Thread(target=gil_10_pc, daemon=True))
    gil_threads[-1].daemon = True
    gil_threads[-1].start()

pipeline.set_state(Gst.State.PLAYING)

if gil_load_enabled:
    gil_load.start()

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
    if gil_load_enabled:
        gil_load.stop()
    for device, detector, dboxes, image_queue in thread_contexts:
        image_queue.put((None, None))
    for inference_thread in inference_threads:
        inference_thread.join()
    finish_time = time.time()

    open(f'logs/{os.path.splitext(sys.argv[0])[0]}.pipeline.dot', 'w').write(
        Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL)
    )
    pipeline.set_state(Gst.State.NULL)
    print(f'FPS: {frames_processed / (finish_time - start_time):.2f}')
    if gil_load_enabled:
        print()
        print(gil_load.format(gil_load.get()))
