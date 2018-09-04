# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:10:26 2017

@author: TIM
"""

from scipy.spatial import ConvexHull
import numpy as np
#points = np.append([[0,2]],[[2,0]],axis=0)
#points = np.append([[0,0],[0,2]],[[2,0]],axis=0)
points = [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0, 0], [0.5,0.5]]
#points = [[-0.5, -0.7, -0.3], [-0.5, 1.5, -0.5], [0.5, -0.5, -0.5], [0, 0, -0.5], [0.5,0.5, -0.5]]
#points = np.random.randn(30, 6)
hull = ConvexHull(points)