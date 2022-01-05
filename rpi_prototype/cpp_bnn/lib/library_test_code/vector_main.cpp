#include <vector>
#include <chrono>
#include <iostream>
#include <cstdint>


enum string_code {
    mix_1_32,
    mix_8_32,
    float32
};

string_code hashit (std::string const& inString) {
    if (inString == "mix_1_32") return mix_1_32;
    if (inString == "mix_8_32") return mix_8_32;
    return float32;
};

int main(int argc, char * argv[]) {
    const int SIZE = atoi(argv[2]); 
    switch (hashit(argv[1])) {
        case mix_1_32:
            {
                std::vector<bool> a(SIZE*SIZE, 1);
                std::vector<float> b(SIZE*SIZE, 1);
                std::vector<float> c(SIZE*SIZE, 0);
                // initialize a to emulate random behaviour
                for (size_t i=0; i<a.size(); i++) {
                    if (i%2 == 0) {
                        a[i] = 0;
                    }
                }

                auto start_2 = std::chrono::high_resolution_clock::now();
                for (size_t i = 0; i<SIZE; i++) {
                    for (size_t j = 0; j<SIZE; j++) {
                        float c_tmp = 0;
                        for (size_t common = 0; common<SIZE; common++) {
                            if (a[i*SIZE + common]) {
                                c_tmp += b[i*SIZE + common];
                            } else {
                                c_tmp -= b[i*SIZE + common];
                            }
                        }
                        c[i*SIZE+j] = c_tmp;
                    }
                }
                auto stop_2 = std::chrono::high_resolution_clock::now();
                printf("Vector [mix_1_32] size: %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
                }
            break;
            
        case mix_8_32:
            {
                
                std::vector<short> a(SIZE*SIZE, 1);
                std::vector<float> b(SIZE*SIZE, 1);
                std::vector<float> c(SIZE*SIZE, 0);
                // initialize a to emulate random behaviour
                for (size_t i=0; i<a.size(); i++) {
                    if (i%2 == 0) {
                        a[i] = 0;
                    }
                }

                auto start_2 = std::chrono::high_resolution_clock::now();
                for (size_t i = 0; i<SIZE; i++) {
                    for (size_t j = 0; j<SIZE; j++) {
                        float c_tmp = 0;
                        for (size_t common = 0; common<SIZE; common++) {
                            c_tmp += (float(a[i*SIZE + common])*b[i*SIZE + common]);
                        }
                        c[i*SIZE+j] = c_tmp;
                    }
                }
                auto stop_2 = std::chrono::high_resolution_clock::now();
                printf("Vector [mix8_32] size: %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case float32:
            {
                
                std::vector<float> a(SIZE*SIZE, 1);
                std::vector<float> b(SIZE*SIZE, 1);
                std::vector<float> c(SIZE*SIZE, 0);
                // initialize a to emulate random behaviour
                for (size_t i=0; i<a.size(); i++) {
                    if (i%2 == 0) {
                        a[i] = 0;
                    }
                }

                auto start_2 = std::chrono::high_resolution_clock::now();
                for (size_t i = 0; i<SIZE; i++) {
                    for (size_t j = 0; j<SIZE; j++) {
                        float c_tmp = 0;
                        for (size_t common = 0; common<SIZE; common++) {
                            c_tmp += (a[i*SIZE + common]*b[i*SIZE + common]);
                        }
                        c[i*SIZE+j] = c_tmp;
                    }
                }
                auto stop_2 = std::chrono::high_resolution_clock::now();
                printf("Vector [float32] size: %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
    }
    return 0;
}