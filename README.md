# lanelet2_extraction

## Installation
Install this library with following command:
```bash
git clone https://github.com/kminoda/lanelet2_extraction.git
cd lanelet2_extraction
python setup.py install
```

## Tutorial
Extract lanelet2

```python
from lanelet2_extraction import Lanelet2Extractor
extractor = Lanelet2Extractor(dataroot='...', maproot='...')
extracted_polylines = extractor.extract(pose, map_name, normalize=False)
```

Project it on an image
```python
from lanelet2_extraction import project_lanelet2_on_image
img = project_lanelet2_on_image(extracted_polylines, img, extrinsic, intrinsic)
```