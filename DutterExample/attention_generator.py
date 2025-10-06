import random
import pickle

TARGET_LENGTH=7

# 10 is addition
# 11 is sos
# 12 is eos

def get_numbers(max_length, num_numbers):
    num_list = [[random.randint(0, 9) for i in range(random.randint(1, max_length))] for j in range(num_numbers)]
    value_list = [int(''.join(map(str, digits))) for digits in num_list]
    return num_list, value_list


def get_addition_list(max_length, num_numbers):
    num_list1, value_list1 = get_numbers(max_length, num_numbers)
    num_list2, value_list2 = get_numbers(max_length, num_numbers)
    num_list = [num_list1[i] + [10] + num_list2[i] for i in range(len(num_list1))]
    value_list = [value_list1[i] + value_list2[i] for i in range(len(num_list1))]
    value_list = [[int(d) for d in str(num)] for num in value_list]
    value_list = [[0] * (TARGET_LENGTH-len(value_list[i])-1) + value_list[i] + [12] for i in range(len(value_list))]
    num_list = [[11] + num_list[i] + [12] for i in range(len(num_list))]
    return num_list, value_list


def sort_addition_list(max_length, num_numbers):
    num_list, value_list = get_addition_list(max_length, num_numbers)
    pairs = list(set([(len(num_list[i]), len(value_list[i])) for i in range(len(num_list))]))
    my_dict = {key: [[num_list[i], value_list[i]] for i in range(len(num_list)) if
                     len(num_list[i]) == key[0] and len(value_list[i]) == key[1]] for key in pairs}
    with open("addition_dict.pkl", 'wb') as f:
        pickle.dump(my_dict, f)


sort_addition_list(5, 500000)
