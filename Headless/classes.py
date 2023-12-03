class Points:
    class Affine:
        def __init__(self, affine_matrix=None):
            self.affine_matrix = affine_matrix

    def __init__(self, image, **kwargs):
        self.data = image.data
        self.name = kwargs.get('name', None)
        self.face_color = kwargs.get('face_color', None)
        self.size = kwargs.get('size', None)
        self.affine = self.Affine(kwargs.get('affine', None))



class Image:
    def __init__(self, image, name):
        self.data = image
        self.name = name
