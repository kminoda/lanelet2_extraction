import numpy as np

class LabeledPolylines:
    def __init__(self):
        self._data = {}

    def add_polyline(self, label, polyline):
        if label in self._data:
            self._data[label].append(polyline)
        else:
            self._data[label] = [polyline]

    def get_polylines(self, label):
        return self._data.get(label, [])
    
    def get_all_labels(self):
        return list(self._data.keys())

    def get_all_polylines(self):
        return list(self._data.values())

    def empty(self):
        return np.sum([len(polylines) for polylines in self._data.values()]) == 0

    def __str__(self):
        return str(self._data)