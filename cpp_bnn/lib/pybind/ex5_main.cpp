#include <iostream>
#include <vector>
#include <stdlib.h>
#include <chrono>

double vsum(const std::vector<double>& vs) {
    double res = 0;
    for (const auto& i: vs) {
        res += i;
    }
    return res;
}

std::vector<double> range(int start, int stop, int step) {
    std::vector<double> res(stop-start);
    for (int i=start; i<stop; i+=step) {
        res[i] = double(i);
    }
    return res;
}


int main(int argc, char * argv[]) {
    auto start = std::chrono::high_resolution_clock::now();
    double res = vsum(range(0, atoi(argv[1]), 1));
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Forward (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
}