import geopandas as gpd
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

FLOWER_PATH = "/workspaces/flowerstrips/data/flowerstripsdata/flowerstrips_BB_2025.gpkg"
TREE_PATH = "/workspaces/flowerstrips/data/trees/tree_canopy_data3035/polygons/Trees12-25_BB.gpkg"
OUT_PATH = "/workspaces/flowerstrips/data/flowerstripsdata/overlaytrees/flowerstrips_overlay.gpkg"

N_WORKERS = 8
CHUNK_SIZE = 300

trees = None

def init_worker(tree_path, target_crs):
    global trees
    trees = gpd.read_file(tree_path)
    trees = trees.to_crs(target_crs)
    trees.sindex

def process_chunk(chunk):
    global trees

    minx, miny, maxx, maxy = chunk.total_bounds
    possible_idx = list(trees.sindex.intersection((minx, miny, maxx, maxy)))

    if possible_idx:
        trees_sub = trees.iloc[possible_idx]
        result = gpd.overlay(chunk, trees_sub, how="difference")
    else:
        result = chunk.copy()

    return result


if __name__ == "__main__":

    flower = gpd.read_file(FLOWER_PATH)

    chunks = [
        flower.iloc[i:i+CHUNK_SIZE].copy()
        for i in range(0, len(flower), CHUNK_SIZE)
    ]

    with Pool(
        processes=N_WORKERS,
        initializer=init_worker,
        initargs=(TREE_PATH, flower.crs)
    ) as pool:

        results = list(
            tqdm(
                pool.imap(process_chunk, chunks),
                total=len(chunks),
                desc="Overlay progress"
            )
        )

    result = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs=flower.crs)

    result.to_file(OUT_PATH, driver="GPKG")

    print("Task completed successfully. The resulting file is saved as 'flowerstrips_overlay.gpkg'.")