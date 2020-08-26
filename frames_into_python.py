
import gi
gi.require_version('Gst', '1.0')

from gi.repository import Gst

Gst.init()

frame_format = 'RGBA'

pipeline = Gst.parse_launch(f'''
    filesrc location=media/in.mp4 !
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
        msg = pipeline.get_bus().timed_pop_filtered(Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        if msg is None: continue

        structure = msg.get_structure()
        print(f'Message from {msg.src.name} [{Gst.message_type_get_name(msg.type)}] {structure.to_string() if structure else ""}')

        if msg.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR):
            break
finally:
    Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, 'DONE')
    pipeline.set_state(Gst.State.NULL)

