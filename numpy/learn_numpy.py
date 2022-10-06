# Learn Numpy

import numpy as np
import pandas as pd

import os

np.set_printoptions(threshold=100)


def np_version():
    print(np.__version__)


# np_version()

# Empty array
def empty_array():
    empty_array = np.empty([2, 2], int)
    print(empty_array)
    print(empty_array.shape)
    print(empty_array.size)


# empty_array()

# 2. Check whether the array is empty
def check_empty():
    a = np.array([])
    b = np.array([1, 2])
    if a.size == 0:
        print(a, 'a : Empty')
    if b.size == 0:
        print('b is empty')


# check_empty()


c = np.array([])
d = np.array([1, 2])


def get_elements(c_array):
    return c_array.ndim and c_array.size


def get_element_count():
    print(c, ', elements_count : ', get_elements(c))
    print(d, ', elements_count : ', get_elements(d))


# get_element_count()

def arange_keyword():
    a = np.arange(4, 12)
    print(a, '.shape: ', a.shape)


# arange_keyword()

e = np.arange(12, 30, 3)
# print(e)
e = e.reshape(2, 3)
# print('After reshaping')
# print(e)


# Random integers
f = np.random.randint(10, size=5)
# print(f)

# Array of Strings
g = np.array(('Toronto', 'Montreal', 'New York'))
# print(g)
# print(g.dtype)


# Numpy to csv
h = np.asarray([[1, 2, 3], [4, 5, 6]])
# print(h)


df = pd.DataFrame({'a1': [1, 2, 3], 'a2': [4, 5, 6]}, index=['X', 'Y', 'Z'])

# print('Dataframe:')
# print(df)

i = df.to_numpy()
# print(i)


j = df.index.to_numpy()
# print(j)

k = df['a1'].to_numpy()


# print(k)

def get_nth_column():
    x = np.array([[1, 2], [3, 4], [5, 6]])

    print('Numpy array:')
    print(x)

    y = x[:, 0]
    print('\nx[:,0]:')
    print(y)

    z = x[:, 1]
    print('\nx[:, 1]:')
    print(z)

    a = x[1, :]
    print('\nx[1,:]:')
    print(a)


# get_nth_column()


def numpy_with_precision():
    # 17. Numpy with precision

    x = np.random.random(10)

    print('Original Array:')
    print(x)

    print('\nAfter setting presicion:')
    np.set_printoptions(precision=2)
    print(x)

    # resetting precision to default (8)
    np.set_printoptions(precision=8)


# numpy_with_precision()


def argsort():
    # 18. Argsort on Numpy array

    a = np.random.randint(0, 10, (3, 3))
    print('Before : ')
    print(a)

    print('\nAfter : ')
    b = a[a[:, 2].argsort()]
    print(b)


# argsort()


def inv():
    b = np.array([[2, 3], [4, 5]])
    print('Before Inverse : ')
    print(b)

    c = np.linalg.inv(b)
    print('After Inverse : ')
    print(c)


# inv()


def np_compare():
    a = np.arange(12).reshape((3, 4))
    print(a)

    a_bool = a < 6
    print(a_bool)


# np_compare()


# Flip numpy array using flipud
def flipud():
    a = np.arange(4).reshape(2, 2)
    print('Before : ')
    print(a)

    print('\nAfter : ')
    b = np.flipud(a)
    print(b)

    # Note: b returns a view
    print('Shared memory? :', np.shares_memory(a, b))


# flipud()


def flipud_without_mem_share():
    a = np.arange(4).reshape(2, 2)
    print('Before : ')
    print(a)

    print('\nAfter : ')
    b = np.flipud(a).copy()
    print(b)

    # Note: b returns a view
    print('Shared memory? :', np.shares_memory(a, b))


# flipud_without_mem_share()

# Convert Numpy array to List
def arr_to_list():
    a = np.arange(10).reshape(2, 5)
    print('Before : ')
    print(a)
    # print(d.dtype)
    print(type(a))
    print(type(a[0]))
    print(type(a[0][0]))

    b = a.tolist()
    print('\nAfter : ')
    print(b)
    print(type(b))
    print(type(b[0]))
    print(type(b[0][0]))


# arr_to_list()


def np_where():
    a = np.arange(8).reshape((2, 4))
    print('Before:')
    print(a)

    print('\nAfter: ')
    b = np.where(a < 4, 0, 20)
    print(b)


# np_where()


def where_mult():
    a = np.arange(8).reshape((2, 4))
    print('Before : ')
    print(a)

    print('\nAfter : ')
    b = np.where((a > 3) & (a < 7), 0, 20)
    print(b)


# where_mult()

# List to numpy array

def list_to_arr():
    a = [1, 2, 3]
    print('Before:')
    print(a)

    b = np.array(a)
    print('After:')
    print(b)

    c = np.array(a, dtype=float)
    print(c)


# list_to_arr()


# 2D list
def list_to_arr2():
    a = [[0, 1, 2], [21, 22, 23]]
    print('Before : ')
    print(a)
    print(type(a))

    print('\nAfter : ')
    b = np.array(a)
    print(b)
    print(type(b))
    print(b.dtype)
    print(b.shape)


# list_to_arr2()

def list_to_float_array():
    x = [1, 2]
    print('Before : ')
    print(x)
    print(type(x))

    print('\nAfter : ')
    b = np.asfarray(x)
    print(b)
    print(type(b))
    print(b.dtype)

    print('\nAfter : ')
    c = np.asarray(x, float)
    print(c)
    print(type(c))
    print(c.dtype)


# list_to_float_array()

# Find common values between 2 arrays

def common_intersect():
    a = np.random.randint(0, 10, 10)
    b = np.random.randint(0, 10, 10)
    print(a)
    print(b)
    print('common values between a and b : ', np.intersect1d(a, b))


# common_intersect()

def today_delta():
    today = np.datetime64('today', 'D')
    print('today          : ', today)

    after2days = np.datetime64('today', 'D') + np.timedelta64(2, 'D')
    print('after 2 days   : ', after2days)

    before3days = np.datetime64('today', 'D') - np.timedelta64(3, 'D')
    print('before 3 days  : ', before3days)

    after1week = np.datetime64('today', 'D') + np.timedelta64(1, 'W')
    print('after 1 week   : ', after1week)

    after10weeks = np.datetime64('today', 'D') + np.timedelta64(10, 'W')
    print('after 10 weeks : ', after10weeks)


# today_delta()

def sort():
    a = np.random.random(5)

    print('Before : ')
    print(a)

    a.sort()
    print('\nAfter : ')
    print(a)


# sort()


def swap_rows():
    a = np.arange(9).reshape(3, 3)
    print(a)

    a[[0, 1]] = a[[1, 0]]
    print(a)


# swap_rows()

def shuffle():
    a = np.arange(20)
    print(a)
    np.random.shuffle(a)
    print(a)


# shuffle()

def get_specific_element():
    a = np.arange(27).reshape(3, 3, 3)
    print(a[0, 1, 1])


# get_specific_element()


def repeat():
    a = np.array([[1, 2, 3]])
    print('Before : ')
    print(a)

    b = np.repeat(a, 3, axis=0)
    print('\nAfter : ')
    print(b)


# repeat()

def min_max_sum():
    a = np.arange(6).reshape(2, 3)
    a += 1
    print(a)

    a_mean = np.min(a)
    print('Mean : ', a_mean)

    a_max = np.max(a)
    print('Max : ', a_max)

    a_sum = np.sum(a)
    print('Sum : ', a_sum)


# min_max_sum()

def get_min_of_axis():
    x = np.arange(10).reshape((2, 5))

    print('x:')
    print(x)

    print('\nx.min(axis = 1) : ')
    print(x.min(axis=1))


# get_min_of_axis()


def get_90_percentile():
    x = np.arange(6).reshape((2, 3))
    print('x:')
    print(x)

    print('\nnp.percentile(x, 90, 0): ')
    print(np.percentile(x, 90, 0))


# get_90_percentile()

def median():
    x = np.arange(6).reshape((2, 3))
    print('x:')
    print(x)

    print('\nnp.median(x): ')
    print(np.median(x))


# median()

def covarience_matrix():
    x = np.array([0, 1, 2])
    y = np.array([7, 8, 9])

    print('x:')
    print(x)
    print('\ny:')
    print(y)

    print('\nnp.cov(x, y): ')
    print(np.cov(x, y))


# covarience_matrix()


def cross_corelate():
    x = np.array([0, 1, 3])
    y = np.array([2, 4, 5])
    print('x:')
    print(x)
    print('\ny:')
    print(y)

    print('\nnp.correlate(x, y): ')


# cross_corelate()


x = np.random.randint(10, 20, (4, 2))

# print('np.random.randint(10, 20, (4, 2)):\n')
# print(x)

y = np.random.choice(40, 4, replace=False)


# print('np.random.choice(40, 4, replace = False):\n')
# print(y)


def permutation():
    a = np.arange(5)
    print(a)
    print('Permutation')
    b = np.random.permutation(a)
    print(b)


# permutation()


def setdiff():
    x = np.array([0, 1, 2, 5, 0])
    y = np.array([0, 1, 4])

    print('x:')
    print(x)

    print('\ny:')
    print(y)

    print('\nnp.setdiff1d(x, y):')
    print(np.setdiff1d(x, y))


# setdiff()

def inv_singular():
    b = np.array([[2, 3], [4, 6]])

    try:
        np.linalg.inv(b)
    except Exception as err:
        print('Error : ', err)


# inv_singular()


def union():
    x = np.array([0, 1, 2, 5, 0])
    y = np.array([0, 1, 4])

    print('x:')
    print(x)

    print('\ny:')
    print(y)

    z = np.union1d(x, y)
    print('\nnp.union1d(x, y):')
    print(z)


# union()

def np_array_to_pd_df():
    x = np.array([[90, 98], [92, 99]])
    print('Numpy Array:')
    print(x)

    df = pd.DataFrame({'Maths': x[:, 0], 'Science': x[:, 1]})
    print('\nDataframe from Numpy Array:')
    print(df)


# np_array_to_pd_df()

def reverse1d():
    x = np.arange(4)
    print('Before:')
    print(x)

    y = x[::-1]
    print('\nAfter reversing:')
    print(y)


# reverse1d()

def reverse2d():
    x = np.arange(8).reshape(2, -1)
    print('Before:')
    print(x)

    y = x[::-1]
    print('\nAfter reversing:')
    print(y)


# reverse2d()

def delete_specific_indices():
    a = np.array([1, 3, 5, 4, 7])
    print('Before:')
    print(a)

    indices = [2, 3]
    b = np.delete(a, indices)

    print('\nAfter Deleting specific indices:')
    print(b)


# delete_specific_indices()


def delete_specific_elements():
    a = np.array([1, 4, 5, 4])
    print('Before:')
    print(a)

    b = np.array([3, 4])
    c = np.setdiff1d(a, b)

    print('\nAfter Deleting specific elements:')
    print(c)


# delete_specific_elements()


a = np.random.rand(2, 4)
# print('Before:')
# print(a)

a[a > 0.5] = 0.5
# print('\nAfter updating a[a > 0.5] = 0.5:')
# print(a)


# Vectorize

aa = np.array([[1,2,3,4], [2,3,4,5], [5,6,7,8], [9,10,11,12]])
bb = np.array([[100,200,300,400], [100,200,300,400], [100,200,300,400], [100,200,300,400]])

def vec2(a, b):
    return a + b

func2 = np.vectorize(vec2)
# print(func2(bb[:,1], aa[:,1]))


# All function calls
np_version()
empty_array()
check_empty()
get_element_count()
arange_keyword()
get_nth_column()
numpy_with_precision()
argsort()
inv()
np_compare()
flipud()
arr_to_list()
np_where()
where_mult()
list_to_arr()
list_to_arr2()
list_to_float_array()
common_intersect()
today_delta()
sort()
swap_rows()
shuffle()
get_specific_element()
repeat()
min_max_sum()
get_min_of_axis()
get_90_percentile()
median()
covarience_matrix()
cross_corelate()
permutation()
setdiff()
inv_singular()
union()
np_array_to_pd_df()
reverse1d()
reverse2d()
delete_specific_indices()
delete_specific_elements()
