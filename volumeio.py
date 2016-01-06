import numpy as np
import struct


def createMitsubaGridVolume(data, file_path, aabb_ext):
    '''Convert raw numpy CT data to mitsuba grid volume format.
    data: numpy type with shape [zdim, ydim, xdim, channels], channels can be 1 or 3. To keep the underlying
          memory layout consistent with C-style indexing, the order 'zyxc' is used.
    file_path: result binary file in mitsuba grid volume format.
    aabb_ext: bounding box in numpy array with shape (2,3), [minx, miny, minz; maxx, maxy, maxz].
    '''
    # See Mitsuba grid-based volume format
    with open(file_path, 'wb') as f:
        f.write('V'), f.write('O'), f.write('L')    # file type flags
        f.write(struct.pack('@B', 3))               # file format version number, currently 3
        f.write(struct.pack('@i', 1))               # encoding identifier, currently only supporting dense float32
        f.write(struct.pack('@iii', data.shape[2], data.shape[1], data.shape[0]))    # xres, yres, zres
        num_channels = 1
        if data.shape[3] == 3:
            num_channels = 3
        f.write(struct.pack('@i', num_channels))    # number of channels
     
        # aabb
        f.write(struct.pack('@3f', aabb_ext[0,0], aabb_ext[0,1], aabb_ext[0,2]))
        f.write(struct.pack('@3f', aabb_ext[1,0], aabb_ext[1,1], aabb_ext[1,2]))

        # volume raw data
        f.write(data.tobytes())    # keep the memory layout of underlying array data consistent with the shape
            

def createMitsubaGridVolumeSimple(data, file_path, half_size=0.5):
    ''' Create a mitsuba volume with aabb bouded in [-half_size, half_size]. Specific axis' size is scaled by data resolution.
    data: [x, y, z, (chanel)].
    half_size: half size of the input data's maxium axis
    ''' 
    # reshape to [x, y, z, c]
    if len(data.shape) == 3:
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))
    # reshape to [z, y, x, c]
    data = data.swapaxes(0, 2)

    # calculate aabb based on input size simply
    xyz_res = np.array([data.shape[2], data.shape[1], data.shape[0]], dtype='float32')
    max_res = np.max(xyz_res)
    xyz_max = half_size * xyz_res / max_res
    xyz_min = - xyz_max

    aabb_ext = np.array([[xyz_min[0], xyz_min[1], xyz_min[2]],
                         [xyz_max[0], xyz_max[1], xyz_max[2]]])

    createMitsubaGridVolume(data, file_path, aabb_ext)


if __name__ == '__main__':
    pass
