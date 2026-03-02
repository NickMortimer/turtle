import  cameratransform as ct

def getpoly(camera:ct.Camera,image_width,image_height,perside=20,crop=0,corners=0):
    import numpy as np
    from shapely.geometry import Polygon
    xl = crop
    xr = image_width - crop
    yt = crop
    yb = image_height - crop
    x = np.linspace(corners, image_width - corners, num=perside)  
    y = np.linspace(corners, image_height - corners, num=perside)
    bottom = np.dstack((x,np.ones(perside)*yb))[0]
    right = np.dstack((np.ones(perside)*xr,y[::-1]))[0]
    top = np.dstack((x[::-1],np.zeros(perside)*yt))[0]
    left = np.dstack((np.ones(perside)*xl,y))[0]
    points = np.vstack((bottom,right,top,left))
    polydata = camera.spaceFromImage(points)
    coords = polydata[:,0:2]
    # Remove any points with NaN or Inf
    #coords = coords[np.isfinite(coords).all(axis=1)]
    # Remove duplicate points
    #coords = np.unique(coords, axis=0)
    poly = Polygon(coords)
    return poly
    # Ensure the ring is closed
    if len(coords) < 3:
        return None  # Not enough points for a polygon
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    try:

        if not poly.is_valid or poly.is_empty:
            return None
        return poly
    except Exception:
        return None
