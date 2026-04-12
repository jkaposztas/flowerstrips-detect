import rasterio

files = [
    "/workspaces/flowerstrips/data/satellite/output_cookie_cuts/april/1107-1376_chip_8_0_vrt_clip.tif",
    "/workspaces/flowerstrips/data/satellite/output_cookie_cuts/june/1107-1376_chip_8_0_vrt_clip.tif",
    "/workspaces/flowerstrips/data/satellite/output_cookie_cuts/august/1107-1376_chip_8_0_vrt_clip.tif",
    "/workspaces/flowerstrips/data/flowerstripsdata/labels/output_cookie_cuts/1107-1376_chip_8_0_vrt_clip.tif"
]

for f in files:
    with rasterio.open(f) as src:
        print(f)
        print("Transform:", src.transform)
        print("Shape:", src.shape)
        print()