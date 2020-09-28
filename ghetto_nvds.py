import torch
from ctypes import Structure, POINTER, CDLL, addressof, sizeof, memmove, byref
from ctypes import c_uint, c_int, c_ulong, c_void_p, c_bool

nvbufsurface = CDLL('libnvbufsurface.so')

max_planes = 4
structure_padding = 4


class NvBufSurfacePlaneParams(Structure):
    _fields_ = [
        ("num_planes", c_uint),
        ("width", c_uint * max_planes),
        ("height", c_uint * max_planes),
        ("pitch", c_uint * max_planes),
        ("offset", c_uint * max_planes),
        ("psize", c_uint * max_planes),
        ("bytesPerPix", c_uint * max_planes),
        ("_reserved", c_void_p * max_planes * structure_padding)
    ]


class NvBufSurfaceMappedAddr(Structure):
    _fields_ = [
        ("addr", c_void_p * max_planes),
        ("eglImage", c_void_p),
        ("_reserved", c_void_p * structure_padding)
    ]


class NvBufSurfaceParams(Structure):
    _fields_ = [
        ("width", c_uint),
        ("height", c_uint),
        ("pitch", c_uint),
        ("colorFormat", c_int),
        ("layout", c_int),
        ("bufferDesc", c_ulong),
        ("dataSize", c_uint),
        ("dataPtr", c_void_p),
        ("planeParams", NvBufSurfacePlaneParams),
        ("mappedAddr", NvBufSurfaceMappedAddr),
        ("_reserved", c_void_p * structure_padding)
    ]


class NvBufSurface(Structure):
    _fields_ = [
        ("gpuId", c_uint),
        ("batchSize", c_uint),
        ("numFilled", c_uint),
        ("isContiguous", c_bool),
        ("memType", c_int),
        ("surfaceList", POINTER(NvBufSurfaceParams)),
        ("_reserved", c_void_p * structure_padding)
    ]

    def __init__(self, gst_map_info):
        nvbufsurface.NvBufSurfaceMemSet(byref(self), -1, -1, 0)
        memmove(addressof(self), gst_map_info.data, min(sizeof(self), len(gst_map_info.data)))

    def struct_copy_from(self, other_buf_surface):
        self.batchSize = other_buf_surface.batchSize
        self.numFilled = other_buf_surface.numFilled
        self.isContiguous = other_buf_surface.isContiguous
        self.memType = other_buf_surface.memType
        self.surfaceList = (NvBufSurfaceParams * other_buf_surface.numFilled)()
        for surface_ix in range(other_buf_surface.numFilled):
            self.surfaceList[surface_ix] = NvBufSurfaceParams()
            self.surfaceList[surface_ix].width = other_buf_surface.surfaceList[surface_ix].width
            self.surfaceList[surface_ix].height = other_buf_surface.surfaceList[surface_ix].height
            self.surfaceList[surface_ix].pitch = other_buf_surface.surfaceList[surface_ix].pitch
            self.surfaceList[surface_ix].colorFormat = other_buf_surface.surfaceList[surface_ix].colorFormat
            self.surfaceList[surface_ix].layout = other_buf_surface.surfaceList[surface_ix].layout
            self.surfaceList[surface_ix].bufferDesc = other_buf_surface.surfaceList[surface_ix].bufferDesc
            self.surfaceList[surface_ix].dataSize = other_buf_surface.surfaceList[surface_ix].dataSize
            self.surfaceList[surface_ix].planeParams = other_buf_surface.surfaceList[surface_ix].planeParams

    def mem_copy_from(self, other_buf_surface):
        copy_result = nvbufsurface.NvBufSurfaceCopy(byref(other_buf_surface), byref(self))
        assert(copy_result == 0)



