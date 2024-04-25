import numpy as np
from stl import mesh


def stl_to_point_cloud(stl_file_path, save_file_path=None):
    # load STL file
    mesh_data = mesh.Mesh.from_file(stl_file_path)

    vertices = np.unique(mesh_data.vectors.reshape((-1, 3)), axis=0)

    # save
    if save_file_path:
        np.savetxt(save_file_path, vertices)
        print("file already saved at:", save_file_path)

    return vertices



stl_file_path = r"F:\MSc_Project_Dataset\eiffel-tower-seetheworld-by-nkabra56\EiffelTowerSeeTheWorld.stl"
save_file_path = r"F:\MSc_Project_Dataset\eiffel-tower-seetheworld-by-nkabra56\EiffelTowerSeeTheWorld_PC.txt"
point_cloud = stl_to_point_cloud(stl_file_path, save_file_path)



