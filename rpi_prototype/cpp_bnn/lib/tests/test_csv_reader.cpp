#include "../utils/data_type.h"
#include "../utils/initialiser.h"
#include "../utils/csv_reader.h"

void test_csv_reader() {
    std::string filename = "/home/ubuntu/imperial_project/BNN/core/cython/lib/tests/test_1.csv";
    
    //std::string filename = "../../lib/tests/test_1.csv";
    
    printf("Offset 0: ");
    Matrix<int> test_data = read_csv<int>(filename, {2,4});
    print_mat(test_data);
    
    printf("Offset 1: ");
    test_data = read_csv<int>(filename, {1,4}, 1);
    print_mat(test_data);
}

int main() {
    //test_csv_reader();
    //std::string filename = "/home/ubuntu/imperial_project/BNN/dataset/csv/mnist/mnist_X_test.csv";
    std::string filename = "/home/ubuntu/imperial_project/BNN/dataset/csv/mnist/mnist_y_train.csv";
    
    //std::string filename = "../../lib/tests/test_1.csv";
    
    Matrix<float> test_data = read_csv<float>(filename, {2,10});
    //Matrix<float> test_data = read_csv<float>(filename, {2,784});
    print_mat(test_data);
    return 0;
}