import numpy as np
import os
import sys


# Directory Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
TRAINING_DATA_DIR = os.path.join(BASE_DIR, 'training_data')
BINARY_DIR = os.path.join(TRAINING_DATA_DIR, 'binary')
BIPOLAR_DIR = os.path.join(TRAINING_DATA_DIR, 'bipolar')
TEST_DIR = os.path.join(BASE_DIR, 'test_data')
BINARY_TEST_PATH = os.path.join(TEST_DIR, 'binary.txt')
BIPOLAR_TEST_PATH = os.path.join(TEST_DIR, 'bipolar.txt')


# Input/Output Settings
CHARACTER_SHAPE = (9, 7)


# Parameters
BINARY_THETA = 0
BIPOLAR_THETA = 0
INPUT_LIMIT = 100


def _get_input_vector(input_file_path):
    input_list = []

    with open(input_file_path, 'r') as input_file:
        for line in input_file.readlines():
            line = line.strip()
            if line:
                input_strings = line.split()
                for input_string in input_strings:
                    input_list.append(int(input_string))

    input_vector = np.zeros((len(input_list), 1))
    for j, input_number in enumerate(input_list):
        input_vector[j, 0] = input_number

    return input_vector


def _get_input_vectors(mode):
    if mode == 'binary':
        training_data_dir = BINARY_DIR
    else:
        training_data_dir = BIPOLAR_DIR

    inputs = 0
    label_names = os.listdir(training_data_dir)
    for label_name in label_names:
        label_dir = os.path.join(training_data_dir, label_name)
        for input_file_name in os.listdir(label_dir):
            input_file_path = os.path.join(label_dir, input_file_name)
            input_vector = _get_input_vector(input_file_path)

            yield input_vector
            inputs += 1
            if inputs >= INPUT_LIMIT:
                return


_binary_activation = np.vectorize(lambda x: 0 if x < BINARY_THETA else (x if x == BINARY_THETA else 1))
_bipolar_activation = np.vectorize(lambda x: -1 if x < BIPOLAR_THETA else (x if x == BIPOLAR_THETA else 1))


def _get_w(input_vectors, mode):
    if mode == 'binary':
        w = sum([np.matmul(2 * input_vector - 1, 2 * input_vector.T - 1) for input_vector in input_vectors])
    else:
        w = sum([np.matmul(input_vector, input_vector.T) for input_vector in input_vectors])
    np.fill_diagonal(w, 0)
    return w


def apply(input_vector, w, mode):
    output_vector = input_vector
    last_output_vector = np.zeros(input_vector.shape)

    while np.any(last_output_vector - output_vector):
        last_output_vector = output_vector

        for i in np.random.permutation(len(output_vector)):
            y_in = input_vector[i] + sum([output_vector[j] * w[j, i] for j in range(len(output_vector))])
            if mode == 'binary':
                output_vector[i] = _binary_activation(y_in)
            else:
                output_vector[i] = _bipolar_activation(y_in)

    return output_vector


def _pretty_print(character_vector):
    reshaped_vector = character_vector.reshape(CHARACTER_SHAPE)
    for i in range(CHARACTER_SHAPE[0]):
        line_string = ''
        for j in range(CHARACTER_SHAPE[1]):
            if reshaped_vector[i, j] == 1:
                line_string += '#  '
            else:
                line_string += '-  '
        line_string += '\n'
        print(line_string)


def run(mode):
    input_vectors = _get_input_vectors(mode)
    w = _get_w(input_vectors, mode)
    if mode == 'binary':
        binary_input_vector = _get_input_vector(BINARY_TEST_PATH)
        output_vector = apply(binary_input_vector, w, mode)
    else:
        bipolar_input_vector = _get_input_vector(BIPOLAR_TEST_PATH)
        output_vector = apply(bipolar_input_vector, w, mode)
    _pretty_print(output_vector)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Invalid Number of Arguments')
        exit()

    if sys.argv[1] == 'binary':
        run('binary')
    elif sys.argv[1] == 'bipolar':
        run('bipolar')
    else:
        print('Invalid Argument')
        exit()
