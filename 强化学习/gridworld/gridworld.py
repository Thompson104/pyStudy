import numpy as np
import sys
from io import StringIO
from gym.envs.toy_text import discrete

UP      = 0
RIGHT   = 1
DOWN    = 2
LEFT    = 3

class GridworldEnv(discrete.DiscreteEnv):
    metadata = {'render.modes':['human','ansi']}
    def __init__(self,shape=[4,4]):
        if (not isinstance(shape,(list,tuple)) or (not len(shape) == 2)):
            raise ValueError('shape 参数 必须是长度为2的list或tuple')
        self.shape = shape

        nS = np.prod(shape) # 计算状态的数量
        nA = 4 # 动作的数量为4，本例为有4个方向

        MAX_Y = shape[0]
        MAX_X =shape[1]

        P = {}

        # 构造状体矩阵，例如
        # [0, 1, 2, 3],
        # [4, 5, 6, 7],
        # [8, 9, 10, 11]
        grid = np.arange(nS).reshape(shape)

        it = np.nditer(grid,flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y,x = it.multi_index
            # 初始化转移概率字典P
            P[s] = {a : [] for a in range(nA)}
            # lambda表达式，即匿名函数
            is_done = lambda s: s == 0 or s == (nS - 1)
            reward = 0.0 if is_done(s) else -1.0

            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:
                # 计算当前状态的，上下左右状态
                ns_up       = s if y == 0 else s - MAX_X
                ns_right    = s if x == (MAX_X -1) else s + 1
                ns_down     = s if y == (MAX_Y -1 ) else s + MAX_X
                ns_left     = s if x == 0 else s -1
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]
            it.iternext()
        # 初始化状态
        isd = np.ones(nS) / nS

        self.P = P
        super(GridworldEnv, self).__init__(nS, nA, P, isd)

        return

    def _render(self,mode='human',close=False):
        if close:
            return

        outfile = StringIO() if mode =='ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid,flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y,x = it.multi_index

            if self.s == s:
                output = 'x'
            elif s== 0 or s == self.nS -1:
                output = 'T'
            else:
                output = 'o'
                pass

            # 判断是否是最左边（x==0）或最右边（x==MaxX -1）
            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] -1 :
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write('\n')


            it.iternext()





