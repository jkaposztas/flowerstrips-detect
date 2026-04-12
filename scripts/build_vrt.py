"""
This code reprojects all the Sentinel2 rasters to the same coordinate system
(3035 suggested). So far, because Brandenburg is in between to UTM zones,
some images have crs for zone 32 and others for zone 33

It then builds a vrt mosaic from the reprojected files.
A vrt is a virtual raster that references the original files using their path.
If you move these files to another folder, the vrt will stop working. 
You can open it with a text editor to see the paths
You could open it in QGIS, but it will be too big for it.
You can test it with only a few raster images first (3 or 4)

 You will use the chips later to clip this vrt into real (i.e. not virtual) rasters
 that you will use for training. 
"""

from osgeo import gdal
import glob, os

# select the paths where your images are, where you will store the reprojected files, 
# and where you will store the vrt called mosaic_repr.vrt in this example
image_dir = '/workspaces/flowerstrips/data/flowerstripsdata/flowerstripsfinal/raster'
# reprojected_dir = '/workspaces/flowerstrips/data/satellite/satellite_data3035/august'
vrt_file = '/workspaces/flowerstrips/data/flowerstripsdata/labels/flowerstrips_repr.vrt'
# target_crs = "EPSG:3035"
reprojected_dir = image_dir # if you want to reproject the CRS comment here and enable the lines above

# create a list with the path to each image inside the image_dir folder
image_list = glob.glob(os.path.join(image_dir, "*.tif"))

# loop through the list and reproject each image
#for tif in image_list:
#    out_tif = os.path.join(reprojected_dir, os.path.basename(tif))
#    ds = gdal.Warp(out_tif, tif, dstSRS=target_crs, dstNodata=0)
#    ds = None  # flush to disk

# Second part. If you have already reprojected the images, you can start from here.
# create another list with the reprojected files
reproj_list = glob.glob(os.path.join(reprojected_dir, "*.tif"))

# Build VRT with absolute paths
vrt_options = gdal.BuildVRTOptions(
    resolution="average",
    addAlpha=True,
    srcNodata=0,
    separate=False,
)
vrt_ds = gdal.BuildVRT(vrt_file, reproj_list, options=vrt_options)

# Flush to disk
vrt_ds = None

print(f"task done, vrt saved to {vrt_file}")
