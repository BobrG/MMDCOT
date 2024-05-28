
### Sk3D
To download main part of the dataset (images, calibration data and depth maps) please follow instructions [here](https://github.com/Skoltech-3D/sk3d_data?tab=readme-ov-file#download). 
You can use config conf.tsv from this folder to download only necessary parts of the dataset.

Once you downloaded main part of the dataset you need to convert depth maps to point clouds. For this use the following command:
```
python generate_points.py /path/to/dir/
```

To generate text data please use ipynb notebook or download predownloaded portion of data [here](https://drive.google.com/file/d/1PSwf8ApCZuunEcnRerku4U8nLaDpolo0/view?usp=sharing). 
For generationg of the text data from scratch you will require OpenAI API key.
