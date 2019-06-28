import numpy as np
import json


class ExtendedJsonEncoder(json.JSONEncoder):

    def default(self, obj):

        if isinstance(obj, np.integer):

            return int(obj)

        elif isinstance(obj, np.floating):

            return float(obj)

        elif isinstance(obj, np.ndarray):

            return obj.tolist()

        else:

            return super(ExtendedJsonEncoder, self).default(obj)
