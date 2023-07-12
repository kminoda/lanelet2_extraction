# lanelet2_extraction

## Installation
Install this library with following command:
```bash
git clone https://github.com/kminoda/lanelet2_extraction.git
cd lanelet2_extraction
python setup.py install
```

## Tutorial
Sample code:
```python
from lanelet2_extraction import Lanelet2Extractor
extractor = Lanelet2Extractor(dataroot='...', maproot='...')
extracted_polylines = extractor.extract(pose, map_name, normalize=False)
```
