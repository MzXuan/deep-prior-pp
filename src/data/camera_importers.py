from importers import DepthImporter
import numpy as np

class CameraImporter(DepthImporter):
    def __init__(self, useCache=True, cacheDir='./cache/', refineNet=None,
                 allJoints=False, hand=None,fx=630.131, fy=630.131, resx=640., resy=480.):
        """
        Constructor
        fx fy: focal length
        resx, resy: resolution
        :return:
        """
        super(CameraImporter, self).__init__(fx, fy, resx/2, resy/2, hand)

        self.depth_map_size = (resx,resy ) # FLAG: notice size!!!
        self.useCache = useCache
        self.cacheDir = cacheDir
        self.allJoints = allJoints
        self.numJoints = 36
        if self.allJoints:
            self.crop_joint_idx = 32
        else:
            self.crop_joint_idx = 13
        self.default_cubes = (250, 250, 250)
        # self.sides = {'train': 'right', 'test_1': 'right', 'test_2': 'right', 'test': 'right', 'train_synth': 'right',
        #               'test_synth_1': 'right', 'test_synth_2': 'right', 'test_synth': 'right'}
        # # joint indices used for evaluation of Tompson et al.
        self.restrictedJointsEval = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]
        self.refineNet = refineNet



    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in xrange(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        ret[0] = (sample[0] - self.ux) * sample[2] / self.fx
        ret[1] = (self.uy - sample[1]) * sample[2] / self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in xrange(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3, ), np.float32)
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = self.uy-sample[1]/sample[2]*self.fy
        ret[2] = sample[2]
        return ret