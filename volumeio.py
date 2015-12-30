# mitsuba volume reader/writer
# TODO: refact below code
import numpy as np
import struct


def createMitsubaGridVolume(data, file_path, bbox_ext=0.5):
    '''Convert numpy ct data to mitsuba grid volume format.

    data: numpy type with shape [zdim, ydim, xdim, channels], channels can be 1 or 3.
    file_path: result binary file in mitsuba grid volume format.
    '''
    assert len(data.shape) == 4, 'invalid data shape'
    assert data.shape[3] == 1 or data.shape[3] == 3, 'invalid channels'

    # See Mitsuba grid-based volume format
    with open(file_path, 'wb') as f:
        f.write('V'), f.write('O'), f.write('L') # file type flags
        f.write(struct.pack('@B', 3))    # file format version number, currently 3
        f.write(struct.pack('@i', 1))    # encoding identifier, currently only supporting dense float32
        f.write(struct.pack('@iii', data.shape[2], data.shape[1], data.shape[0]))    # xres, yres, zres
        num_channels = 1
        if data.shape[3] == 3:
            num_channels = 3
        f.write(struct.pack('@i', num_channels))    # number of channels
     
        # calculate Axis-aligned bounding box
        xyz_res = np.array([data.shape[2], data.shape[1], data.shape[0]], dtype='float32')
        max_res = np.max(xyz_res)
        xyz_max = bbox_ext * xyz_res / max_res
        xyz_min = -xyz_max

        f.write(struct.pack('@3f', xyz_min[0], xyz_min[1], xyz_min[2]))
        f.write(struct.pack('@3f', xyz_max[0], xyz_max[1], xyz_max[2]))
#        f.write(struct.pack('@6f', xmin, ymin, zmin, xmax, ymax, zmax))

        # volume raw data
        f.write(data.tobytes())    # keep the memory layout of underlying array data consistent with the shape
            

def createSilkDensityVolume():
    src = r'D:\Dataset\silk\silk_dn.dat'
    dst = r'D:\Dataset\silk\silk_density.vol'
    data = np.load(src)

    data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))

    # reshape (105, 1013, 992, 1) to (105, 992, 1013, 1)
    data = np.swapaxes(data, 1, 2)

    createMitsubaGridVolume(data, dst)


def createSilkOrientationVolume():
    src = r'D:\Dataset\silk\orientation_dn.dat'
    dst = r'D:\Dataset\silk\silk_orientation.vol'

    data = np.load(src) # alreay float32

    # reshape (105, 1013, 992, 3) to (105, 992, 1013, 3)
    data = np.swapaxes(data, 1, 2)
    createMitsubaGridVolume(data, dst)


def createVolume(src, dst):
    data = np.load(src) # float32, [z, x, y, c]

    if len(data.shape) == 3:
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))

    data = np.swapaxes(data, 1, 2)
    print data.shape

    createMitsubaGridVolume(data, dst)


def createVolume(src, dst, sz, ez, bbox_ext=0.5):
    data = np.load(src) # float32, [z, x, y, c]
    data = data[sz:ez]

    if len(data.shape) == 3:
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))

    data = np.swapaxes(data, 1, 2)
    print data.shape

    createMitsubaGridVolume(data, dst, bbox_ext)


def createSilkVolume():
    src = r'D:\Dataset\silk\silk_dn.dat'
    dst = r'D:\Dataset\silk\silk_dn_z64104_b01_density.vol'
    bbox_ext = 0.1
    createVolume(src, dst, 64, 104, bbox_ext)
    print 'create density finished ...'

    src = r'D:\Dataset\silk\orientation_dn.dat'
    dst = r'D:\Dataset\silk\silk_dn_z64104_b01_orientation.vol'
    createVolume(src, dst, 64, 104, bbox_ext)
    print 'create orientation finished ...'


def creaetSimpleTitledSilkVolume(src, dst):
    data = np.load(src)
    if len(data.shape) == 3:
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))
    data = np.swapaxes(data, 1, 2)

    sz = 64
    ez = 104
    sy = 200
    ey = 400
    sx = 600
    ex = 800
    bbox_ext = 0.1

    data = data[sz:ez, sy:ey, sx:ex]
    print data.shape

    nz = 2
    ny = 10
    nx = 8

    data = np.tile(data, (nz, ny, nx, 1))
    print data.shape

    createMitsubaGridVolume(data, dst, bbox_ext)


def createTitleVolume():
    src = r'D:\Dataset\silk\silk_dn.dat'
    dst = r'D:\Dataset\silk\silk_dn_title2x10x8_b01_density.vol'
    creaetSimpleTitledSilkVolume(src, dst) 
    print 'finish density ...'

    src = r'D:\Dataset\silk\orientation_dn.dat'
    dst = r'D:\Dataset\silk\silk_dn_title2x10x8_b01_orientation.vol'
    creaetSimpleTitledSilkVolume(src, dst) 
    print 'finish orientation ...'


def createFeltVolume1():
    # density
    sz = 200
    ez = 508
    bbox_ext = 0.25
    felt_den_file = r'D:\Dataset\felt\felt_dn.dat'
    den_vol_file = r'D:\Dataset\felt\density200_508.vol'

    data = np.load(felt_den_file)   # [508, 1013 992]
    data = data.swapaxes(1, 2)  #[508, 992, 1013]
    data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))
    data = data[sz:ez]

    print data.shape
    createMitsubaGridVolume(data, den_vol_file, bbox_ext)
    del data

    # orienation
    ori_vol_file = r'D:\Dataset\felt\orientation200_508.vol'

    ori2 = r'D:\Dataset\felt\ori200_300_dn.dat'
    ori3 = r'D:\Dataset\felt\ori300_400_dn.dat'
    ori4 = r'D:\Dataset\felt\ori400_500_dn.dat'
    ori5 = r'D:\Dataset\felt\ori500_508_dn.dat'

    ori2 = np.load(ori2)  # [100, 1013, 992]
    ori3 = np.load(ori3)
    ori4 = np.load(ori4)
    ori5 = np.load(ori5)

    ori = np.concatenate((ori2, ori3, ori4, ori5), axis=0)
    ori = ori.swapaxes(1, 2)

    del ori2, ori3, ori4, ori5

    print ori.shape
    createMitsubaGridVolume(ori, ori_vol_file, bbox_ext)


if __name__ == '__main__':
    pass
