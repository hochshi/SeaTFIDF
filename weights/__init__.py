from idfweights import *
from tfweights import *

recommended_weighting_schemes = [
    {
        'doc': {
            'tf': 'raw_count',
            'idf': 'idf'
        },
        'query': {
            'tf': 'double_norm',
            'idf': 'idf'
        }
    },
    {
        'doc': {
            'tf': 'log_norm',
            'idf': 'unary'
        },
        'query': {
            'tf': 'binary',
            'idf': 'idf_smooth'
        }
    },
    {
        'doc': {
            'tf': 'log_norm',
            'idf': 'idf'
        },
        'query': {
            'tf': 'log_norm',
            'idf': 'idf'
        }
    }
]