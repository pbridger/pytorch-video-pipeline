import os, sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

frame_format = 'RGBA'

Gst.init()
pipeline = Gst.parse_launch(f'''
    filesrc location=media/in.mp4 num-buffers=200 !
    decodebin !
    fakesink name=s
''')

def on_frame_probe(pad, info):
    buf = info.get_buffer()
    print(f'[{buf.pts / Gst.SECOND:6.2f}]')
    return Gst.PadProbeReturn.OK

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
