"""A recursive repr method for classes that contain memory addresses in their repr."""


def recursive_repr(obj):
    """A recursive repr method for classes that contain memory addresses in their repr."""
    if hasattr(obj, '__dict__'):
        # Start constructing the representation string based on the class name
        repr_str = f'{obj.__class__.__name__}('
        first = True
        for key, value in obj.__dict__.items():
            if not first:
                repr_str += ', '
            first = False
            # Recursively apply the same logic to nested objects
            if key == 'sound_file_paths':
                # Removes the full paths. 
                # Assuming the path was given as data/raw/year/audio thats what is appended to the repr
                truncated_value = '/'.join(value[0].split('/')[-5:-1])
                repr_str += f'{key}={truncated_value}'
            else:
                if hasattr(value, '__dict__'):
                    repr_str += f'{key}={recursive_repr(value)}'
                else:
                    repr_str += f'{key}={value}'
        repr_str += ')'
        return repr_str
    else:
        # If no __dict__, return the default __repr__
        return repr(obj)