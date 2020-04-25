''' YAML configurable transformation pipeline '''

import albumentations as A


def Transpose(permutation=(2, 0, 1)):
    ''' Transpose input data, expecting PyTorch axis transformation 
    
    Args:
      permutation: A list of int specifying permutation rule.

    Returns:
      An albumentation object.
    '''
    def f(x):
        return x.transpose(*permutation)

    return A.Lambda(image=f, mask=f)


ADDONS = {'Transpose': Transpose}


def build_transform(names, params):
    ''' Build data transformation pipeline

    Args:
      names: A list of str, indicating operation class.
      params: A dict of dict, specifing operation arguments.

    Returns:
      albumentations.Compose object.
    '''
    transforms = []
    for name in names:
        if hasattr(A, name):
            T = getattr(A, name)
        elif name in ADDONS.keys():
            T = ADDONS[name]
        else:
            raise KeyError(f'{name} transformation is not implemented')

        # Build instance with/without arguments
        transform = T(**params[name]) if name in params.keys() else T()
        transforms.append(transform)
    return A.Compose(transforms)
