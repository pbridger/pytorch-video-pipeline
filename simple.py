
import os, sys
import gi
gi.require_version('Gst', '1.0')

from gi.repository import Gst

Gst.init()

pipeline = Gst.parse_launch('''
    filesrc location=media/in.mp4 !
    decodebin !
    fakesink name=s
''')


def on_buffer(pad, info):
    print(f'[{info.get_buffer().pts / Gst.SECOND:6.2f}]')
    return Gst.PadProbeReturn.OK

pipeline.get_by_name('s').get_static_pad('sink').add_probe(Gst.PadProbeType.BUFFER, on_buffer)


pipeline.set_state(Gst.State.PLAYING)

while True:
    message = pipeline.get_bus().timed_pop_filtered(Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
    if message is None: continue

    structure = message.get_structure()
    print('Message:', message.src.name, Gst.message_type_get_name(message.type), structure.to_string() if structure else '')

    if message.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR):
        break

Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, 'EXIT')
pipeline.set_state(Gst.State.NULL)
