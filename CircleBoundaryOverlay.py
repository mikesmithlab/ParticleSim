from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np



# This helper function projects a point from 3d space to 
# 2d window coordinates.
def project_point(xyz, painter, args):
    view_tm = args['view_tm'] # 3x4 matrix
    proj_tm = args['proj_tm'] # 4x4 matrix
    world_pos = np.append(xyz, 1) # Convert to 4-vector.
    view_pos = np.dot(view_tm, world_pos) # Transform to view space.	
    # Check if point is behind the viewer. If yes, stop here.
    if args['is_perspective'] and view_pos[2] >= 0.0: return None
    # Project to screen space:
    screen_pos = np.dot(proj_tm, np.append(view_pos, 1)) 
    screen_pos[0:3] /= screen_pos[3]
    win_rect = painter.window()
    x = win_rect.left() + win_rect.width() * (screen_pos[0] + 1) / 2
    y = win_rect.bottom() - win_rect.height() * (screen_pos[1] + 1) / 2 + 1
    return (x,y)	

# This helper function projects a distance or radius from 3d space to 
# 2d window coordinates.
def project_radius(xyz, r, painter, args):
    if args['is_perspective']:
        world_pos = np.append(xyz, 1) # Convert to 4-vector.
        vp = np.append(np.dot(args['view_tm'], world_pos), 1) # Transform to view space.	
        p1 = np.dot(args['proj_tm'], vp) # Project to screen space.
        p1[0:3] /= p1[3]
        vp += [0,r,0,0]
        p2 = np.dot(args['proj_tm'], vp) # Project to screen space.
        p2[0:3] /= p2[3]
        return np.linalg.norm(p2-p1) * painter.window().height() / 2
    else:
        return r / args['fov'] * painter.window().height() / 2



# This function is called by OVITO on every viewport update.
def render(painter, **args):
    if args['is_perspective']: 
        raise Exception("This only works with non-perspective viewports.")
    # Define geometry of circular boundary in screen space
    RingRad = 11.25#mm
    origin = (0,0,0)
    circle_color = QColor(0,0,255)#Blue
	
	#Convert to screen scale etc
    RingRadScaled = project_radius(origin, RingRad, painter, args)
    origin_screen = project_point(origin, painter, args)
    diameter = RingRadScaled*2
   
    #Draw circle on top
    rect = QRectF(origin_screen[0]-RingRadScaled, origin_screen[1]-RingRadScaled, diameter, diameter)
    painter.setPen(QPen(circle_color));
    painter.drawEllipse(rect)