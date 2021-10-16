import numpy as np

def getdata1():

    X = [[12, 7, 3],
         [4, 5, 6],
         [7, 8, 9]]


    Y = [[5, 8 ],
         [6, 7 ],
         [4, 5 ]]

    return X,Y

def getdata2():

    X2 = [[12, 7, 3],
         [4, 5, 6],
         [7, 8, 9]]


    Y2 = [[5, 8, 4 ],
         [6, 7, 5 ],
         [4, 5, 6 ]]

    return X2,Y2

def mat_mul(X,Y):

    result = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in X]

    return result

def mat_mul_elem(X,Y):
    """Element by element multiplication
    Returns list.
    Inputs X and Y has to be same dimension
    """
    return np.multiply(X,Y).tolist()

def three_d_array(value, *dim):
    """
    Create 3D-array
    :param dim: a tuple of dimensions - (x, y, z)
    :param value: value with which 3D-array is to be filled
    :return: 3D-array
    """

    return [[[value for _ in range(dim[2])] for _ in range(dim[1])] for _ in range(dim[0])]


def main():

    X,Y = getdata1()

    res = mat_mul(X, Y)
    print(res)

    X2,Y2 = getdata2()

    res2 = mat_mul_elem(X2, Y2)
    print(res2)

    print(three_d_array("hi", *(2,3,3)))


if __name__ == '__main__':

    main()
