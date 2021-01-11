# coding = 'utf-8'
import numpy as np
import pandas as pd
import time

def target_mean_v1(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        groupby_result = data[data.index != i].groupby([x_name], as_index=False).agg(['mean', 'count'])
        result[i] = groupby_result.loc[groupby_result.index == data.loc[i, x_name], (y_name, 'mean')]
    return result


def target_mean_v2(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result


def main():

    y = np.random.randint(2, size=(500, 1))
    x = np.random.randint(10, size=(500, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])

    start = time.time()
    result_1 = target_mean_v1(data, 'y', 'x')
    end = time.time()
    print('v1 time cost:', end - start)

    start = time.time()
    result_2 = target_mean_v2(data, 'y', 'x')
    end = time.time()
    print('v1 time cost:', end - start)

    diff = np.linalg.norm(result_1 - result_2)
    print('diff = ', diff)

    from target_mean_cy import target_mean_v3, target_mean_v4
    matrix = data.to_numpy(dtype=np.int_)

    start = time.time()
    result_3 = target_mean_v3(matrix)
    end = time.time()
    print('v3_cy time cost:', end - start)

    diff = np.linalg.norm(result_2 - result_3)
    print('diff = ', diff)

    start = time.time()
    result_4 = target_mean_v4(matrix, data.shape[0], 10)
    end = time.time()
    print('v4_cy time cost:', end - start)

    diff = np.linalg.norm(result_2 - result_4)
    print('diff = ', diff)

    # import timeit
    # print(timeit.timeit("test()", setup="from __main__ import test"))


if __name__ == '__main__':
    main()
