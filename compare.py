import numpy as np

def compare_values(array1,array2, num_bins=10, num_highest_values=2):
    array1 = array1.flatten()
    array2 = array2.flatten()
    if len (array1) != len (array2):
        print("arrays differ")
        return
    diff = np.abs(array1 - array2)
    mean_diff = np.mean(diff)
    median_diff = np.median(diff)
    histogram_diff = np.histogram(diff, num_bins)
    print ("the mean differences are", mean_diff)
    print ("the median differences are", median_diff)
    print("the histogram bucket interval is", histogram_diff[1])
    print("the histogram diff count in each bucket is", histogram_diff[0])
    top_differences_indexes =  diff.argsort()[-num_highest_values:]
    difference_value = np.take(diff, top_differences_indexes)
    array1_difference_value = np.take(array1, top_differences_indexes)
    array2_difference_value = np.take(array2, top_differences_indexes)
    print ("the arguments with top differences are below: columns are array index, value in array1, value in array2, difference")
    print(np.array2string(np.column_stack((top_differences_indexes, array1_difference_value, array2_difference_value, difference_value)),separator=',').replace(' [ ','').replace('],', '').strip('[ ]'))


a = np.array([1,2,15,25, 4,54,455])
b = np.array([8, 1,2,15,25,54,455])
compare_values(a,b)
