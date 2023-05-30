__author__ = 'mhgou'

import numpy as np
import copy
from grasp_nms import nms_grasp

GRASP_ARRAY_LEN = 17
RECT_GRASP_ARRAY_LEN = 7
EPS = 1e-8

class Grasp():
    def __init__(self, *args):
        '''
        **Input:**

        - args can be a numpy array or tuple of the score, width, height, depth, rotation_matrix, translation, object_id

        - the format of numpy array is [score, width, height, depth, rotation_matrix(9), translation(3), object_id]

        - the length of the numpy array is 17.
        '''
        if len(args) == 0:
            self.grasp_array = np.array([0, 0.02, 0.02, 0.02, 1, 0, 0, 0, 1 ,0 , 0, 0, 1, 0, 0, 0, -1], dtype = np.float64)
        elif len(args) == 1:
            if type(args[0]) == np.ndarray:
                self.grasp_array = copy.deepcopy(args[0])
            else:
                raise TypeError('if only one arg is given, it must be np.ndarray.')
        elif len(args) == 7:
            score, width, height, depth, rotation_matrix, translation, object_id = args
            self.grasp_array = np.concatenate([np.array((score, width, height, depth)),rotation_matrix.reshape(-1), translation, np.array((object_id)).reshape(-1)]).astype(np.float64)
        else:
            raise ValueError('only 1 or 7 arguments are accepted')
    
    def __repr__(self):
        return 'Grasp: score:{}, width:{}, height:{}, depth:{}, translation:{}\nrotation:\n{}\nobject id:{}'.format(self.score, self.width, self.height, self.depth, self.translation, self.rotation_matrix, self.object_id)

    @property
    def score(self):
        '''
        **Output:**

        - float of the score.
        '''
        return float(self.grasp_array[0])

    @score.setter
    def score(self, score):
        '''
        **input:**

        - float of the score.
        '''
        self.grasp_array[0] = score

    @property
    def width(self):
        '''
        **Output:**

        - float of the width.
        '''
        return float(self.grasp_array[1])
    
    @width.setter
    def width(self, width):
        '''
        **input:**

        - float of the width.
        '''
        self.grasp_array[1] = width

    @property
    def height(self):
        '''
        **Output:**

        - float of the height.
        '''
        return float(self.grasp_array[2])

    @height.setter
    def height(self, height):
        '''
        **input:**

        - float of the height.
        '''
        self.grasp_array[2] = height
    
    @property
    def depth(self):
        '''
        **Output:**

        - float of the depth.
        '''
        return float(self.grasp_array[3])

    @depth.setter
    def depth(self, depth):
        '''
        **input:**

        - float of the depth.
        '''
        self.grasp_array[3] = depth

    @property
    def rotation_matrix(self):
        '''
        **Output:**

        - np.array of shape (3, 3) of the rotation matrix.
        '''
        return self.grasp_array[4:13].reshape((3,3))

    @rotation_matrix.setter
    def rotation_matrix(self, *args):
        '''
        **Input:**

        - len(args) == 1: tuple of matrix

        - len(args) == 9: float of matrix
        '''
        if len(args) == 1:
            self.grasp_array[4:13] = np.array(args[0],dtype = np.float64).reshape(9)
        elif len(args) == 9:
            self.grasp_array[4:13] = np.array(args,dtype = np.float64)

    @property
    def translation(self):
        '''
        **Output:**

        - np.array of shape (3,) of the translation.
        '''
        return self.grasp_array[13:16]

    @translation.setter
    def translation(self, *args):
        '''
        **Input:**

        - len(args) == 1: tuple of x, y, z

        - len(args) == 3: float of x, y, z
        '''
        if len(args) == 1:
            self.grasp_array[13:16] = np.array(args[0],dtype = np.float64)
        elif len(args) == 3:
            self.grasp_array[13:16] = np.array(args,dtype = np.float64)

    @property
    def object_id(self):
        '''
        **Output:**

        - int of the object id that this grasp grasps
        '''
        return int(self.grasp_array[16])

    @object_id.setter
    def object_id(self, object_id):
        '''
        **Input:**

        - int of the object_id.
        '''
        self.grasp_array[16] = object_id

    def transform(self, T):
        '''
        **Input:**

        - T: np.array of shape (4, 4)
        
        **Output:**

        - Grasp instance after transformation, the original Grasp will also be changed.
        '''
        rotation = T[:3,:3]
        translation = T[:3,3]
        self.translation = np.dot(rotation, self.translation.reshape((3,1))).reshape(-1) + translation
        self.rotation_matrix = np.dot(rotation, self.rotation_matrix)
        return self

class GraspGroup():
    def __init__(self, *args):
        '''
        **Input:**

        - args can be (1) nothing (2) numpy array of grasp group array (3) str of the npy file.
        '''
        if len(args) == 0:
            self.grasp_group_array = np.zeros((0, GRASP_ARRAY_LEN), dtype=np.float64)
        elif len(args) == 1:
            if isinstance(args[0], np.ndarray):
                self.grasp_group_array = args[0]
            elif isinstance(args[0], str):
                self.grasp_group_array = np.load(args[0])
            else:
                raise ValueError('args must be nothing, numpy array or string.')
        else:
            raise ValueError('args must be nothing, numpy array or string.')

    def __len__(self):
        '''
        **Output:**

        - int of the length.
        '''
        return len(self.grasp_group_array)

    def __repr__(self):
        repr = '----------\nGrasp Group, Number={}:\n'.format(self.__len__())
        if self.__len__() <= 6:
            for grasp_array in self.grasp_group_array:
                repr += Grasp(grasp_array).__repr__() + '\n'
        else:
            for i in range(3):
                repr += Grasp(self.grasp_group_array[i]).__repr__() + '\n'
            repr += '......\n'
            for i in range(3):
                repr += Grasp(self.grasp_group_array[-(3-i)]).__repr__() + '\n'
        return repr + '----------'

    def __getitem__(self, index):
        '''
        **Input:**

        - index: int, slice, list or np.ndarray.

        **Output:**

        - if index is int, return Grasp instance.

        - if index is slice, np.ndarray or list, return GraspGroup instance.
        '''
        if type(index) == int:
            return Grasp(self.grasp_group_array[index])
        elif type(index) == slice:
            graspgroup = GraspGroup()
            graspgroup.grasp_group_array = copy.deepcopy(self.grasp_group_array[index])
            return graspgroup
        elif type(index) == np.ndarray:
            return GraspGroup(self.grasp_group_array[index])
        elif type(index) == list:
            return GraspGroup(self.grasp_group_array[index])
        else:
            raise TypeError('unknown type "{}" for calling __getitem__ for GraspGroup'.format(type(index)))

    @property
    def scores(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the scores.
        '''
        return self.grasp_group_array[:,0]
    
    @scores.setter
    def scores(self, scores):
        '''
        **Input:**

        - scores: numpy array of shape (-1, ) of the scores.
        '''
        assert scores.size == len(self)
        self.grasp_group_array[:,0] = copy.deepcopy(scores)

    @property
    def widths(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the widths.
        '''
        return self.grasp_group_array[:,1]
    
    @widths.setter
    def widths(self, widths):
        '''
        **Input:**

        - widths: numpy array of shape (-1, ) of the widths.
        '''
        assert widths.size == len(self)
        self.grasp_group_array[:,1] = copy.deepcopy(widths)

    @property
    def heights(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the heights.
        '''
        return self.grasp_group_array[:,2]

    @heights.setter
    def heights(self, heights):
        '''
        **Input:**

        - heights: numpy array of shape (-1, ) of the heights.
        '''
        assert heights.size == len(self)
        self.grasp_group_array[:,2] = copy.deepcopy(heights)

    @property
    def depths(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the depths.
        '''
        return self.grasp_group_array[:,3]

    @depths.setter
    def depths(self, depths):
        '''
        **Input:**

        - depths: numpy array of shape (-1, ) of the depths.
        '''
        assert depths.size == len(self)
        self.grasp_group_array[:,3] = copy.deepcopy(depths)

    @property
    def rotation_matrices(self):
        '''
        **Output:**

        - np.array of shape (-1, 3, 3) of the rotation matrices.
        '''
        return self.grasp_group_array[:, 4:13].reshape((-1, 3, 3))

    @rotation_matrices.setter
    def rotation_matrices(self, rotation_matrices):
        '''
        **Input:**

        - rotation_matrices: numpy array of shape (-1, 3, 3) of the rotation_matrices.
        '''
        assert rotation_matrices.shape == (len(self), 3, 3)
        self.grasp_group_array[:,4:13] = copy.deepcopy(rotation_matrices.reshape((-1, 9)))       

    @property
    def translations(self):
        '''
        **Output:**

        - np.array of shape (-1, 3) of the translations.
        '''
        return self.grasp_group_array[:, 13:16]

    @translations.setter
    def translations(self, translations):
        '''
        **Input:**

        - translations: numpy array of shape (-1, 3) of the translations.
        '''
        assert translations.shape == (len(self), 3)
        self.grasp_group_array[:,13:16] = copy.deepcopy(translations)

    @property
    def object_ids(self):
        '''
        **Output:**

        - numpy array of shape (-1, ) of the object ids.
        '''
        return self.grasp_group_array[:,16]

    @object_ids.setter
    def object_ids(self, object_ids):
        '''
        **Input:**

        - object_ids: numpy array of shape (-1, ) of the object_ids.
        '''
        assert object_ids.size == len(self)
        self.grasp_group_array[:,16] = copy.deepcopy(object_ids)

    def transform(self, T):
        '''
        **Input:**

        - T: np.array of shape (4, 4)
        
        **Output:**

        - GraspGroup instance after transformation, the original GraspGroup will also be changed.
        '''
        rotation = T[:3,:3]
        translation = T[:3,3]
        self.translations = np.dot(rotation, self.translations.T).T + translation # (-1, 3)
        self.rotation_matrices = np.matmul(rotation, self.rotation_matrices).reshape((-1, 3, 3)) # (-1, 9)
        return self

    def add(self, element):
        '''
        **Input:**

        - element: Grasp instance or GraspGroup instance.
        '''
        if isinstance(element, Grasp):
            self.grasp_group_array = np.concatenate((self.grasp_group_array, element.grasp_array.reshape((-1, GRASP_ARRAY_LEN))))
        elif isinstance(element, GraspGroup):
            self.grasp_group_array = np.concatenate((self.grasp_group_array, element.grasp_group_array))
        else:
            raise TypeError('Unknown type:{}'.format(element))
        return self

    def remove(self, index):
        '''
        **Input:**

        - index: list of the index of grasp
        '''
        self.grasp_group_array = np.delete(self.grasp_group_array, index, axis = 0)
        return self

    def from_npy(self, npy_file_path):
        '''
        **Input:**

        - npy_file_path: string of the file path.
        '''
        self.grasp_group_array = np.load(npy_file_path)
        return self

    def save_npy(self, npy_file_path):
        '''
        **Input:**

        - npy_file_path: string of the file path.
        '''
        np.save(npy_file_path, self.grasp_group_array)

    def to_open3d_geometry_list(self):
        '''
        **Output:**

        - list of open3d.geometry.Geometry of the grippers.
        '''
        geometry = []
        for i in range(len(self.grasp_group_array)):
            g = Grasp(self.grasp_group_array[i])
            geometry.append(g.to_open3d_geometry())
        return geometry
    
    def sort_by_score(self, reverse = False):
        '''
        **Input:**

        - reverse: bool of order, if False, from high to low, if True, from low to high.

        **Output:**

        - no output but sort the grasp group.
        '''
        score = self.grasp_group_array[:,0]
        index = np.argsort(score)
        if not reverse:
            index = index[::-1]
        self.grasp_group_array = self.grasp_group_array[index]
        return self

    def random_sample(self, numGrasp = 20):
        '''
        **Input:**

        - numGrasp: int of the number of sampled grasps.

        **Output:**

        - GraspGroup instance of sample grasps.
        '''
        if numGrasp > self.__len__():
            raise ValueError('Number of sampled grasp should be no more than the total number of grasps in the group')
        shuffled_grasp_group_array = copy.deepcopy(self.grasp_group_array)
        np.random.shuffle(shuffled_grasp_group_array)
        shuffled_grasp_group = GraspGroup()
        shuffled_grasp_group.grasp_group_array = copy.deepcopy(shuffled_grasp_group_array[:numGrasp])
        return shuffled_grasp_group

    def nms(self, translation_thresh = 0.03, rotation_thresh = 30.0 / 180.0 * np.pi):
        '''
        **Input:**

        - translation_thresh: float of the translation threshold.

        - rotation_thresh: float of the rotation threshold.

        **Output:**

        - GraspGroup instance after nms.
        '''
        return GraspGroup(nms_grasp(self.grasp_group_array, translation_thresh, rotation_thresh))
