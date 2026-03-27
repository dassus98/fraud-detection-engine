import numpy as np
import logging

logger = logging.getLogger(__name__)


def reduce_mem_usage(df, verbose=True):
    """Downcast numeric columns to the smallest safe type to reduce DataFrame memory footprint."""
    numerics = ['int16', 'int32', 'int64', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        # Only process numeric columns
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            # Integer handling
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

            # Float handling — stops at float32 to prevent aggregation precision loss
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.info(
            'Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df
