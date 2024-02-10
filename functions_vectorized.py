def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Vectorized implementation.
    """
    x2 = x.diagonal()
    x2 = x2[x2 != 0]
    ans = x2.prod()
    return ans

def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Vectorized implementation.
    """
    x2 = np.unique(x)
    y2 = np.unique(y)
    return len(np.setxor1d(x2, y2)) == 0

def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Vectorized implementation.
    """
    ans = -1e18
    y = np.arange(0, len(x), 1)
    y = y[x[y] == 0]
    y = y[y < len(x) - 1]
    y += 1
    y = x[y]
    if len(y) > 0:
        ans = y.max()
    return ans

def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Vectorized implementation.
    """
    img = np.dot(img, coefs)
    return img

def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Vectorized implementation.
    """
    ans = np.unique(x, return_counts=True)
    return ans


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Vctorized implementation.
    """
    return np.sqrt(np.sum((x[:, np.newaxis] - y) ** 2, axis = -1))