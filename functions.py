def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Not vectorized implementation.
    """
    ans = 1
    for i in range(min(len(x), len(x[0]))):
        if x[i][i] != 0:
            ans *= x[i][i]
    return ans


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Not vectorized implementation.
    """
    x2 = set(x)
    y2 = set(y)
    return x2 == y2


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Not vectorized implementation.
    """
    ans = -1e18
    for i in range(1, len(x)):
        if x[i - 1] == 0:
            ans = max(ans, x[i])
    return ans

def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """
    img2 = [[0] * len(img[0]) for _ in range(len(img))]
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(len(coefs)):
                img2[i][j] += coefs[k] * img[i][j][k]
    return img2


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Not vectorized implementation.
    """
    cnt_num = dict()
    for el in x:
        cnt_num[el] = 0
    for el in x:
        cnt_num[el] += 1
    ans1, ans2 = [], []
    for el in cnt_num:
        ans1.append(el)
        ans2.append(cnt_num[el])
    return [ans1, ans2]   


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Not vectorized implementation.
    """
    n = len(x)
    m = len(y)
    ans = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(len(x[i])):
                ans[i][j] += (x[i][k] - y[j][k]) ** 2
            ans[i][j] = ans[i][j] ** 0.5
    return ans