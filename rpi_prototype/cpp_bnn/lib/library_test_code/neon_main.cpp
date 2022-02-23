/*
 * Copyright (C) Arm Limited, 2019 All rights reserved. 
 * 
 * The example code is provided to you as an aid to learning when working 
 * with Arm-based technology, including but not limited to programming tutorials. 
 * Arm hereby grants to you, subject to the terms and conditions of this Licence, 
 * a non-exclusive, non-transferable, non-sub-licensable, free-of-charge licence, 
 * to use and copy the Software solely for the purpose of demonstration and 
 * evaluation.
 * 
 * You accept that the Software has not been tested by Arm therefore the Software 
 * is provided "as is", without warranty of any kind, express or implied. In no 
 * event shall the authors or copyright holders be liable for any claim, damages 
 * or other liability, whether in action or contract, tort or otherwise, arising 
 * from, out of or in connection with the Software or the use of Software.
 */
//https://developer.arm.com/architectures/instruction-sets/intrinsics/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <vector>

#include <chrono>
#include <arm_neon.h>
#include <string>



void matrix_multiply_c(float32_t *A, float32_t *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k) {
    for (int i_idx=0; i_idx<n; i_idx++) {
        for (int j_idx=0; j_idx<m; j_idx++) {
            C[n*j_idx + i_idx] = 0;
            for (int k_idx=0; k_idx<k; k_idx++) {
                C[n*j_idx + i_idx] += A[n*k_idx + i_idx]*B[k*j_idx + k_idx];
            }
        }
    }
}

void matrix_multiply_c(float16_t *A, float16_t *B, float16_t *C, uint32_t n, uint32_t m, uint32_t k) {
    for (int i_idx=0; i_idx<n; i_idx++) {
        for (int j_idx=0; j_idx<m; j_idx++) {
            C[n*j_idx + i_idx] = 0;
            for (int k_idx=0; k_idx<k; k_idx++) {
                C[n*j_idx + i_idx] += A[n*k_idx + i_idx]*B[k*j_idx + k_idx];
            }
        }
    }
}

void matrix_multiply_c(int16_t *A, int16_t *B, int16_t *C, uint32_t n, uint32_t m, uint32_t k) {
    for (int i_idx=0; i_idx<n; i_idx++) {
        for (int j_idx=0; j_idx<m; j_idx++) {
            C[n*j_idx + i_idx] = 0;
            for (int k_idx=0; k_idx<k; k_idx++) {
                C[n*j_idx + i_idx] += A[n*k_idx + i_idx]*B[k*j_idx + k_idx];
            }
        }
    }
}
void matrix_multiply_c(int8_t *A, int8_t *B, int16_t *C, uint32_t n, uint32_t m, uint32_t k) {
    for (int i_idx=0; i_idx<n; i_idx++) {
        for (int j_idx=0; j_idx<m; j_idx++) {
            C[n*j_idx + i_idx] = 0;
            for (int k_idx=0; k_idx<k; k_idx++) {
                C[n*j_idx + i_idx] += int16_t (A[n*k_idx + i_idx]*B[k*j_idx + k_idx]);
            }
        }
    }
}
void matrix_multiply_c(int8_t *A, float32_t *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k) {
    for (int i_idx=0; i_idx<n; i_idx++) {
        for (int j_idx=0; j_idx<m; j_idx++) {
            C[n*j_idx + i_idx] = 0;
            for (int k_idx=0; k_idx<k; k_idx++) {
                C[n*j_idx + i_idx] += int32_t (A[n*k_idx + i_idx])*B[k*j_idx + k_idx];
            }
        }
    }
}

void matrix_multiply_neon(float32_t  *A, float32_t  *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k) {
    /* 
     * Multiply matrices A and B, store the result in C. 
     * It is the user's responsibility to make sure the matrices are compatible.
     */     

    int A_idx;
    int B_idx;
    int C_idx;

    // these are the columns of a 4x4 sub matrix of A
    float32x4_t A0;
    float32x4_t A1;
    float32x4_t A2;
    float32x4_t A3;

    // these are the columns of a 4x4 sub matrix of B
    float32x4_t B0;
    float32x4_t B1;
    float32x4_t B2;
    float32x4_t B3;

    // these are the columns of a 4x4 sub matrix of C
    float32x4_t C0;
    float32x4_t C1;
    float32x4_t C2;
    float32x4_t C3;

    for (int i_idx=0; i_idx<n; i_idx+=4) {
        for (int j_idx=0; j_idx<m; j_idx+=4) {
            // Zero accumulators before matrix op
            C0 = vmovq_n_f32(0);
            C1 = vmovq_n_f32(0);
            C2 = vmovq_n_f32(0);
            C3 = vmovq_n_f32(0);
            for (int k_idx=0; k_idx<k; k_idx+=4) {
                // Compute base index to 4x4 block
                A_idx = i_idx + n*k_idx;
                B_idx = k*j_idx + k_idx;

                // Load most current A values in row 
                A0 = vld1q_f32(A+A_idx);
                A1 = vld1q_f32(A+A_idx+n);
                A2 = vld1q_f32(A+A_idx+2*n);
                A3 = vld1q_f32(A+A_idx+3*n);

                // Multiply accumulate in 4x1 blocks, i.e. each column in C
                B0 = vld1q_f32(B+B_idx);
                C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
                C0 = vfmaq_laneq_f32(C0, A1, B0, 1);
                C0 = vfmaq_laneq_f32(C0, A2, B0, 2);
                C0 = vfmaq_laneq_f32(C0, A3, B0, 3);

                B1 = vld1q_f32(B+B_idx+k);
                C1 = vfmaq_laneq_f32(C1, A0, B1, 0);
                C1 = vfmaq_laneq_f32(C1, A1, B1, 1);
                C1 = vfmaq_laneq_f32(C1, A2, B1, 2);
                C1 = vfmaq_laneq_f32(C1, A3, B1, 3);

                B2 = vld1q_f32(B+B_idx+2*k);
                C2 = vfmaq_laneq_f32(C2, A0, B2, 0);
                C2 = vfmaq_laneq_f32(C2, A1, B2, 1);
                C2 = vfmaq_laneq_f32(C2, A2, B2, 2);
                C2 = vfmaq_laneq_f32(C2, A3, B2, 3);

                B3 = vld1q_f32(B+B_idx+3*k);
                C3 = vfmaq_laneq_f32(C3, A0, B3, 0);
                C3 = vfmaq_laneq_f32(C3, A1, B3, 1);
                C3 = vfmaq_laneq_f32(C3, A2, B3, 2);
                C3 = vfmaq_laneq_f32(C3, A3, B3, 3);
            }
            // Compute base index for stores
            C_idx = n*j_idx + i_idx;
            vst1q_f32(C+C_idx, C0);
            vst1q_f32(C+C_idx+n, C1);
            vst1q_f32(C+C_idx+2*n, C2);
            vst1q_f32(C+C_idx+3*n, C3);
        }
    }
}

void matrix_multiply_neon(int16_t  *A, int16_t  *B, int16_t *C, uint32_t n, uint32_t m, uint32_t k) {
    /* 
     * Multiply matrices A and B, store the result in C. 
     * It is the user's responsibility to make sure the matrices are compatible.
     */     

    int A_idx;
    int B_idx;
    int C_idx;

    // these are the columns of a 4x4 sub matrix of A
    int16x8_t A0;
    int16x8_t A1;
    int16x8_t A2;
    int16x8_t A3;

    // these are the columns of a 4x4 sub matrix of B
    int16x8_t B0;
    int16x8_t B1;
    int16x8_t B2;
    int16x8_t B3;

    // these are the columns of a 4x4 sub matrix of C
    int16x8_t C0;
    int16x8_t C1;
    int16x8_t C2;
    int16x8_t C3;

    for (int i_idx=0; i_idx<n; i_idx+=4) {
        for (int j_idx=0; j_idx<m; j_idx+=4) {
            // Zero accumulators before matrix op
            C0 = vmovq_n_s16(0);
            C1 = vmovq_n_s16(0);
            C2 = vmovq_n_s16(0);
            C3 = vmovq_n_s16(0);
            for (int k_idx=0; k_idx<k; k_idx+=4) {
                // Compute base index to 4x4 block
                A_idx = i_idx + n*k_idx;
                B_idx = k*j_idx + k_idx;

                // Load most current A values in row 
                A0 = vld1q_s16(A+A_idx);
                A1 = vld1q_s16(A+A_idx+n);
                A2 = vld1q_s16(A+A_idx+2*n);
                A3 = vld1q_s16(A+A_idx+3*n);

                // Multiply accumulate in 4x1 blocks, i.e. each column in C
                B0 = vld1q_s16(B+B_idx);
                C0 = vmlaq_laneq_s16(C0, A0, B0, 0);
                C0 = vmlaq_laneq_s16(C0, A1, B0, 1);
                C0 = vmlaq_laneq_s16(C0, A2, B0, 2);
                C0 = vmlaq_laneq_s16(C0, A3, B0, 3);

                B1 = vld1q_s16(B+B_idx+k);
                C1 = vmlaq_laneq_s16(C1, A0, B1, 0);
                C1 = vmlaq_laneq_s16(C1, A1, B1, 1);
                C1 = vmlaq_laneq_s16(C1, A2, B1, 2);
                C1 = vmlaq_laneq_s16(C1, A3, B1, 3);

                B2 = vld1q_s16(B+B_idx+2*k);
                C2 = vmlaq_laneq_s16(C2, A0, B2, 0);
                C2 = vmlaq_laneq_s16(C2, A1, B2, 1);
                C2 = vmlaq_laneq_s16(C2, A2, B2, 2);
                C2 = vmlaq_laneq_s16(C2, A3, B2, 3);

                B3 = vld1q_s16(B+B_idx+3*k);
                C3 = vmlaq_laneq_s16(C3, A0, B3, 0);
                C3 = vmlaq_laneq_s16(C3, A1, B3, 1);
                C3 = vmlaq_laneq_s16(C3, A2, B3, 2);
                C3 = vmlaq_laneq_s16(C3, A3, B3, 3);
            }
            // Compute base index for stores
            C_idx = n*j_idx + i_idx;
            vst1q_s16(C+C_idx, C0);
            vst1q_s16(C+C_idx+n, C1);
            vst1q_s16(C+C_idx+2*n, C2);
            vst1q_s16(C+C_idx+3*n, C3);
        }
    }
}

void matrix_multiply_neon(int32_t  *A, int32_t  *B, int32_t *C, uint32_t n, uint32_t m, uint32_t k) {
    /* 
     * Multiply matrices A and B, store the result in C. 
     * It is the user's responsibility to make sure the matrices are compatible.
     */     

    int A_idx;
    int B_idx;
    int C_idx;

    // these are the columns of a 4x4 sub matrix of A
    int32x4_t A0;
    int32x4_t A1;
    int32x4_t A2;
    int32x4_t A3;

    // these are the columns of a 4x4 sub matrix of B
    int32x4_t B0;
    int32x4_t B1;
    int32x4_t B2;
    int32x4_t B3;

    // these are the columns of a 4x4 sub matrix of C
    int32x4_t C0;
    int32x4_t C1;
    int32x4_t C2;
    int32x4_t C3;

    for (int i_idx=0; i_idx<n; i_idx+=4) {
        for (int j_idx=0; j_idx<m; j_idx+=4) {
            // Zero accumulators before matrix op
            C0 = vmovq_n_s32(0);
            C1 = vmovq_n_s32(0);
            C2 = vmovq_n_s32(0);
            C3 = vmovq_n_s32(0);
            for (int k_idx=0; k_idx<k; k_idx+=4) {
                // Compute base index to 4x4 block
                A_idx = i_idx + n*k_idx;
                B_idx = k*j_idx + k_idx;

                // Load most current A values in row 
                A0 = vld1q_s32(A+A_idx);
                A1 = vld1q_s32(A+A_idx+n);
                A2 = vld1q_s32(A+A_idx+2*n);
                A3 = vld1q_s32(A+A_idx+3*n);

                // Multiply accumulate in 4x1 blocks, i.e. each column in C
                B0 = vld1q_s32(B+B_idx);
                C0 = vmlaq_laneq_s32(C0, A0, B0, 0);
                C0 = vmlaq_laneq_s32(C0, A1, B0, 1);
                C0 = vmlaq_laneq_s32(C0, A2, B0, 2);
                C0 = vmlaq_laneq_s32(C0, A3, B0, 3);

                B1 = vld1q_s32(B+B_idx+k);
                C1 = vmlaq_laneq_s32(C1, A0, B1, 0);
                C1 = vmlaq_laneq_s32(C1, A1, B1, 1);
                C1 = vmlaq_laneq_s32(C1, A2, B1, 2);
                C1 = vmlaq_laneq_s32(C1, A3, B1, 3);

                B2 = vld1q_s32(B+B_idx+2*k);
                C2 = vmlaq_laneq_s32(C2, A0, B2, 0);
                C2 = vmlaq_laneq_s32(C2, A1, B2, 1);
                C2 = vmlaq_laneq_s32(C2, A2, B2, 2);
                C2 = vmlaq_laneq_s32(C2, A3, B2, 3);

                B3 = vld1q_s32(B+B_idx+3*k);
                C3 = vmlaq_laneq_s32(C3, A0, B3, 0);
                C3 = vmlaq_laneq_s32(C3, A1, B3, 1);
                C3 = vmlaq_laneq_s32(C3, A2, B3, 2);
                C3 = vmlaq_laneq_s32(C3, A3, B3, 3);
            }
            // Compute base index for stores
            C_idx = n*j_idx + i_idx;
            vst1q_s32(C+C_idx, C0);
            vst1q_s32(C+C_idx+n, C1);
            vst1q_s32(C+C_idx+2*n, C2);
            vst1q_s32(C+C_idx+3*n, C3);
        }
    }
}

void matrix_multiply_neon(int8_t  *A, int8_t  *B, int16_t *C, uint32_t n, uint32_t m, uint32_t k) {
    /* 
     * Multiply matrices A and B, store the result in C. 
     * It is the user's responsibility to make sure the matrices are compatible.
     */     

    int A_idx;
    int B_idx;
    int C_idx;

    // these are the columns of a 4x4 sub matrix of A
    int16x8_t A0;
    int16x8_t A1;
    int16x8_t A2;
    int16x8_t A3;

    // these are the columns of a 4x4 sub matrix of B
    int16x8_t B0;
    int16x8_t B1;
    int16x8_t B2;
    int16x8_t B3;

    // these are the columns of a 4x4 sub matrix of C
    int16x8_t C0;
    int16x8_t C1;
    int16x8_t C2;
    int16x8_t C3;

    for (int i_idx=0; i_idx<n; i_idx+=4) {
        for (int j_idx=0; j_idx<m; j_idx+=4) {
            // Zero accumulators before matrix op
            C0 = vmovq_n_s16(0);
            C1 = vmovq_n_s16(0);
            C2 = vmovq_n_s16(0);
            C3 = vmovq_n_s16(0);
            for (int k_idx=0; k_idx<k; k_idx+=4) {
                // Compute base index to 4x4 block
                A_idx = i_idx + n*k_idx;
                B_idx = k*j_idx + k_idx;

                // Load most current A values in row 
                A0 = vld1q_s16(reinterpret_cast<int16_t*>(A+A_idx));
                A1 = vld1q_s16(reinterpret_cast<int16_t*>(A+A_idx+n));
                A2 = vld1q_s16(reinterpret_cast<int16_t*>(A+A_idx+2*n));
                A3 = vld1q_s16(reinterpret_cast<int16_t*>(A+A_idx+3*n));

                // Multiply accumulate in 4x1 blocks, i.e. each column in C
                B0 = vld1q_s16(reinterpret_cast<int16_t*>(B+B_idx));
                C0 = vmlaq_laneq_s16(C0, A0, B0, 0);
                C0 = vmlaq_laneq_s16(C0, A1, B0, 1);
                C0 = vmlaq_laneq_s16(C0, A2, B0, 2);
                C0 = vmlaq_laneq_s16(C0, A3, B0, 3);

                B1 = vld1q_s16(reinterpret_cast<int16_t*>(B+B_idx+k));
                C1 = vmlaq_laneq_s16(C1, A0, B1, 0);
                C1 = vmlaq_laneq_s16(C1, A1, B1, 1);
                C1 = vmlaq_laneq_s16(C1, A2, B1, 2);
                C1 = vmlaq_laneq_s16(C1, A3, B1, 3);

                B2 = vld1q_s16(reinterpret_cast<int16_t*>(B+B_idx+2*k));
                C2 = vmlaq_laneq_s16(C2, A0, B2, 0);
                C2 = vmlaq_laneq_s16(C2, A1, B2, 1);
                C2 = vmlaq_laneq_s16(C2, A2, B2, 2);
                C2 = vmlaq_laneq_s16(C2, A3, B2, 3);

                B3 = vld1q_s16(reinterpret_cast<int16_t*>(B+B_idx+3*k));
                C3 = vmlaq_laneq_s16(C3, A0, B3, 0);
                C3 = vmlaq_laneq_s16(C3, A1, B3, 1);
                C3 = vmlaq_laneq_s16(C3, A2, B3, 2);
                C3 = vmlaq_laneq_s16(C3, A3, B3, 3);
            }
            // Compute base index for stores
            C_idx = n*j_idx + i_idx;
            vst1q_s16(C+C_idx, C0);
            vst1q_s16(C+C_idx+n, C1);
            vst1q_s16(C+C_idx+2*n, C2);
            vst1q_s16(C+C_idx+3*n, C3);
        }
    }
}

void matrix_multiply_neon(int8_t  *A, float32_t  *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k) {
    /* 
     * Multiply matrices A and B, store the result in C. 
     * It is the user's responsibility to make sure the matrices are compatible.
     */     

    int A_idx;
    int B_idx;
    int C_idx;

    // these are the columns of a 4x4 sub matrix of A
    float32x4_t A0;
    float32x4_t A1;
    float32x4_t A2;
    float32x4_t A3;

    // these are the columns of a 4x4 sub matrix of B
    float32x4_t B0;
    float32x4_t B1;
    float32x4_t B2;
    float32x4_t B3;

    // these are the columns of a 4x4 sub matrix of C
    float32x4_t C0;
    float32x4_t C1;
    float32x4_t C2;
    float32x4_t C3;

    for (int i_idx=0; i_idx<n; i_idx+=4) {
        for (int j_idx=0; j_idx<m; j_idx+=4) {
            // Zero accumulators before matrix op
            C0 = vmovq_n_f32(0);
            C1 = vmovq_n_f32(0);
            C2 = vmovq_n_f32(0);
            C3 = vmovq_n_f32(0);
            for (int k_idx=0; k_idx<k; k_idx+=4) {
                // Compute base index to 4x4 block
                A_idx = i_idx + n*k_idx;
                B_idx = k*j_idx + k_idx;

                // Load most current A values in row 
                A0 = vld1q_f32(reinterpret_cast<float32_t*>(A+A_idx));
                A1 = vld1q_f32(reinterpret_cast<float32_t*>(A+A_idx+n));
                A2 = vld1q_f32(reinterpret_cast<float32_t*>(A+A_idx+2*n));
                A3 = vld1q_f32(reinterpret_cast<float32_t*>(A+A_idx+3*n));

                // Multiply accumulate in 4x1 blocks, i.e. each column in C
                B0 = vld1q_f32(B+B_idx);
                C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
                C0 = vfmaq_laneq_f32(C0, A1, B0, 1);
                C0 = vfmaq_laneq_f32(C0, A2, B0, 2);
                C0 = vfmaq_laneq_f32(C0, A3, B0, 3);

                B1 = vld1q_f32(B+B_idx+k);
                C1 = vfmaq_laneq_f32(C1, A0, B1, 0);
                C1 = vfmaq_laneq_f32(C1, A1, B1, 1);
                C1 = vfmaq_laneq_f32(C1, A2, B1, 2);
                C1 = vfmaq_laneq_f32(C1, A3, B1, 3);

                B2 = vld1q_f32(B+B_idx+2*k);
                C2 = vfmaq_laneq_f32(C2, A0, B2, 0);
                C2 = vfmaq_laneq_f32(C2, A1, B2, 1);
                C2 = vfmaq_laneq_f32(C2, A2, B2, 2);
                C2 = vfmaq_laneq_f32(C2, A3, B2, 3);

                B3 = vld1q_f32(B+B_idx+3*k);
                C3 = vfmaq_laneq_f32(C3, A0, B3, 0);
                C3 = vfmaq_laneq_f32(C3, A1, B3, 1);
                C3 = vfmaq_laneq_f32(C3, A2, B3, 2);
                C3 = vfmaq_laneq_f32(C3, A3, B3, 3);
            }
            // Compute base index for stores
            C_idx = n*j_idx + i_idx;
            vst1q_f32(C+C_idx, C0);
            vst1q_f32(C+C_idx+n, C1);
            vst1q_f32(C+C_idx+2*n, C2);
            vst1q_f32(C+C_idx+3*n, C3);
        }
    }
}
void matrix_multiply_4x4_neon(float32_t *A, float32_t *B, float32_t *C) {
        // these are the columns A
        float32x4_t A0;
        float32x4_t A1;
        float32x4_t A2;
        float32x4_t A3;
        
        // these are the columns B
        float32x4_t B0;
        float32x4_t B1;
        float32x4_t B2;
        float32x4_t B3;
        
        // these are the columns C
        float32x4_t C0;
        float32x4_t C1;
        float32x4_t C2;
        float32x4_t C3;
        
        A0 = vld1q_f32(A);
        A1 = vld1q_f32(A+4);
        A2 = vld1q_f32(A+8);
        A3 = vld1q_f32(A+12);
        
        // Zero accumulators for C values
        C0 = vmovq_n_f32(0);
        C1 = vmovq_n_f32(0);
        C2 = vmovq_n_f32(0);
        C3 = vmovq_n_f32(0);
        
        // Multiply accumulate in 4x1 blocks, i.e. each column in C
        B0 = vld1q_f32(B);
        C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
        C0 = vfmaq_laneq_f32(C0, A1, B0, 1);
        C0 = vfmaq_laneq_f32(C0, A2, B0, 2);
        C0 = vfmaq_laneq_f32(C0, A3, B0, 3);
        vst1q_f32(C, C0);
        
        B1 = vld1q_f32(B+4);
        C1 = vfmaq_laneq_f32(C1, A0, B1, 0);
        C1 = vfmaq_laneq_f32(C1, A1, B1, 1);
        C1 = vfmaq_laneq_f32(C1, A2, B1, 2);
        C1 = vfmaq_laneq_f32(C1, A3, B1, 3);
        vst1q_f32(C+4, C1);
        
        B2 = vld1q_f32(B+8);
        C2 = vfmaq_laneq_f32(C2, A0, B2, 0);
        C2 = vfmaq_laneq_f32(C2, A1, B2, 1);
        C2 = vfmaq_laneq_f32(C2, A2, B2, 2);
        C2 = vfmaq_laneq_f32(C2, A3, B2, 3);
        vst1q_f32(C+8, C2);
        
        B3 = vld1q_f32(B+12);
        C3 = vfmaq_laneq_f32(C3, A0, B3, 0);
        C3 = vfmaq_laneq_f32(C3, A1, B3, 1);
        C3 = vfmaq_laneq_f32(C3, A2, B3, 2);
        C3 = vfmaq_laneq_f32(C3, A3, B3, 3);
        vst1q_f32(C+12, C3);
}

void print_matrix(float32_t *M, uint32_t cols, uint32_t rows) {
        for (int i=0; i<rows; i++) {
                for (int j=0; j<cols; j++) {
                        printf("%f ", M[j*rows + i]);
                }
                printf("\n");
        }
        printf("\n");
}

void matrix_init_rand(float32_t *M, uint32_t numvals) {
        for (int i=0; i<numvals; i++) {
                M[i] = (float)rand()/(float)(RAND_MAX);
        }
}
void matrix_init_rand(float16_t *M, uint32_t numvals) {
        for (int i=0; i<numvals; i++) {
                M[i] = (float)rand()/(float)(RAND_MAX);
        }
}
void matrix_init_rand(int32_t *M, uint32_t numvals) {
        for (int i=0; i<numvals; i++) {
                M[i] = (int32_t)((float)rand()/(float)(RAND_MAX));
        }
}
void matrix_init_rand(int16_t *M, uint32_t numvals) {
        for (int i=0; i<numvals; i++) {
                M[i] = (int16_t)((float)rand()/(float)(RAND_MAX));
        }
}
void matrix_init_rand(int8_t *M, uint32_t numvals) {
        for (int i=0; i<numvals; i++) {
                M[i] = (int8_t)((float)rand()/(float)(RAND_MAX));
        }
}


void matrix_init(int16_t *M, uint32_t cols, uint32_t rows, int16_t val) {
        for (int i=0; i<rows; i++) {
                for (int j=0; j<cols; j++) {
                        M[j*rows + i] = val;
                }
        }
}

void matrix_init(float32_t *M, uint32_t cols, uint32_t rows, float32_t val) {
        for (int i=0; i<rows; i++) {
                for (int j=0; j<cols; j++) {
                        M[j*rows + i] = val;
                }
        }
}

bool f32comp_noteq(float32_t a, float32_t b) {
        if (fabs(a-b) < 0.000001) {
                return false;
        }
        return true;
}

bool matrix_comp(float32_t *A, float32_t *B, uint32_t rows, uint32_t cols) {
        float32_t a;
        float32_t b;
        for (int i=0; i<rows; i++) {
                for (int j=0; j<cols; j++) {
                        a = A[rows*j + i];
                        b = B[rows*j + i];      
                        
                        if (f32comp_noteq(a, b)) {
                                printf("i=%d, j=%d, A=%f, B=%f\n", i, j, a, b);
                                return false;
                        }
                }
        }
        return true;
}

bool matrix_comp(int16_t *A, int16_t *B, uint32_t rows, uint32_t cols) {
        float32_t a;
        float32_t b;
        for (int i=0; i<rows; i++) {
                for (int j=0; j<cols; j++) {
                        a = A[rows*j + i];
                        b = B[rows*j + i];      
                        
                        if (f32comp_noteq(a, b)) {
                                printf("i=%d, j=%d, A=%f, B=%f\n", i, j, a, b);
                                return false;
                        }
                }
        }
        return true;
}
enum string_code {
    int16_stack,
    float32_stack,
    int16_stack_naive,
    float32_stack_naive,
    int16,
    int32,
    float32,
    int16_naive,
    float16_naive,
    float32_naive,
    mix8_16,
    mix8_32,
    check_int16,
    check_float32,
    check_mix8_16,
    check_mix8_32
};

string_code hashit (std::string const& inString) {
    if (inString == "int16_stack") return int16_stack;
    if (inString == "float32_stack") return float32_stack;
    if (inString == "int16_stack_naive") return int16_stack_naive;
    if (inString == "float32_stack_naive") return float32_stack_naive;
    if (inString == "int16") return int16;
    if (inString == "int32") return int32;
    if (inString == "float32") return float32;
    if (inString == "int16_naive") return int16_naive;
    if (inString == "float32_naive") return float32_naive;
    if (inString == "float16_naive") return float16_naive;
    if (inString == "mix8_16") return mix8_16;
    if (inString == "mix8_32") return mix8_32;
    if (inString == "check_int16") return check_int16;
    if (inString == "check_float32") return check_float32;
    if (inString == "check_mix8_16") return check_mix8_16;
    //mix_8_32
    return check_mix8_32;
};
int main(int argc, char * argv[]) {
    const int SIZE = atoi(argv[2]); 
    std::chrono::high_resolution_clock::time_point start_2, stop_2, start, stop;
    uint32_t n = SIZE; // rows in A
    uint32_t m = SIZE; // cols in B
    uint32_t k = SIZE; // cols in a and rows in b
    bool c_eq_asm;
    bool c_eq_neon;
    switch (hashit(argv[1])) {
        case int16_stack:
            {
            int16_t A[n*k];
            int16_t B[k*m];
            int16_t D[n*m];

            matrix_init_rand(A, n*k);
            matrix_init_rand(B, k*m);

            start_2 = std::chrono::high_resolution_clock::now();
            matrix_multiply_neon(&A[0], &B[0], &D[0], n, m, k);
            stop_2 = std::chrono::high_resolution_clock::now();
            printf("Neon %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case float32_stack:
            {
            float32_t A[n*k];
            float32_t B[k*m];
            float32_t D[n*m];

            matrix_init_rand(A, n*k);
            matrix_init_rand(B, k*m);
            
            start_2 = std::chrono::high_resolution_clock::now();
            matrix_multiply_neon(&A[0], &B[0], &D[0], n, m, k);
            stop_2 = std::chrono::high_resolution_clock::now();
            printf("Neon %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
        case int16_stack_naive:
            {
            int16_t A[n*k];
            int16_t B[k*m];
            int16_t D[n*m];

            matrix_init_rand(A, n*k);
            matrix_init_rand(B, k*m);
            
            start_2 = std::chrono::high_resolution_clock::now();
            matrix_multiply_c(&A[0], &B[0], &D[0], n, m, k);
            stop_2 = std::chrono::high_resolution_clock::now();
            printf("Neon %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case float32_stack_naive:
            {
            float32_t A[n*k];
            float32_t B[k*m];
            float32_t D[n*m];

            matrix_init_rand(A, n*k);
            matrix_init_rand(B, k*m);
            
            start_2 = std::chrono::high_resolution_clock::now();
            matrix_multiply_c(&A[0], &B[0], &D[0], n, m, k);
            stop_2 = std::chrono::high_resolution_clock::now();
            printf("Neon %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case int16:
            {
            std::vector<int16_t> A(n*k);
            std::vector<int16_t> B(k*m);
            std::vector<int16_t> D(n*m);
                
            matrix_init_rand(&A[0], n*k);
            matrix_init_rand(&B[0], k*m);

            start_2 = std::chrono::high_resolution_clock::now();
            matrix_multiply_neon(&A[0], &B[0], &D[0], n, m, k);
            stop_2 = std::chrono::high_resolution_clock::now();
            printf("Neon %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case int32:
            {
            std::vector<int32_t> A(n*k);
            std::vector<int32_t> B(k*m);
            std::vector<int32_t> D(n*m);
                
            matrix_init_rand(&A[0], n*k);
            matrix_init_rand(&B[0], k*m);

            start_2 = std::chrono::high_resolution_clock::now();
            matrix_multiply_neon(&A[0], &B[0], &D[0], n, m, k);
            stop_2 = std::chrono::high_resolution_clock::now();
            printf("Neon %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case float32:
            {
            std::vector<float32_t> Af(n*k);
            std::vector<float32_t> Bf(k*m);
            std::vector<float32_t> Df(n*m);
                
            matrix_init_rand(&Af[0], n*k);
            matrix_init_rand(&Bf[0], k*m);

            start_2 = std::chrono::high_resolution_clock::now();
            matrix_multiply_neon(&Af[0], &Bf[0], &Df[0], n, m, k);
            stop_2 = std::chrono::high_resolution_clock::now();
            printf("Neon [float32] size: %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case int16_naive:
            {
            std::vector<int16_t> A(n*k);
            std::vector<int16_t> B(k*m);

            matrix_init_rand(&A[0], n*k);
            matrix_init_rand(&B[0], k*m);
                
            std::vector<int16_t> E(n*m);
            start = std::chrono::high_resolution_clock::now();
            matrix_multiply_c(&A[0], &B[0], &E[0], n, m, k);
            stop = std::chrono::high_resolution_clock::now();
            printf("Neon %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case float16_naive:
            {
            std::vector<float16_t> Af(n*k);
            std::vector<float16_t> Bf(k*m);
            matrix_init_rand(&Af[0], n*k);
            matrix_init_rand(&Bf[0], k*m);

            //float32_t E[n*m];
            std::vector<float16_t> Ef(n*m);
            start = std::chrono::high_resolution_clock::now();
            //matrix_multiply_c(A, B, E, n, m, k);
            matrix_multiply_c(&Af[0], &Bf[0], &Ef[0], n, m, k);
            stop = std::chrono::high_resolution_clock::now();
            printf("Neon %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
            }
            break;
            
        case float32_naive:
            {
            std::vector<float32_t> Af(n*k);
            std::vector<float32_t> Bf(k*m);
            matrix_init_rand(&Af[0], n*k);
            matrix_init_rand(&Bf[0], k*m);

            //float32_t E[n*m];
            std::vector<float32_t> Ef(n*m);
            start = std::chrono::high_resolution_clock::now();
            //matrix_multiply_c(A, B, E, n, m, k);
            matrix_multiply_c(&Af[0], &Bf[0], &Ef[0], n, m, k);
            stop = std::chrono::high_resolution_clock::now();
            printf("Neon %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case mix8_16:
            {
            std::vector<int8_t> Af(n*k);
            std::vector<int8_t> Bf(k*m);
            std::vector<int16_t> Df(n*m);

            matrix_init_rand(&Af[0], n*k);
            matrix_init_rand(&Bf[0], k*m);

            start_2 = std::chrono::high_resolution_clock::now();
            matrix_multiply_neon(&Af[0], &Bf[0], &Df[0], n, m, k);
            stop_2 = std::chrono::high_resolution_clock::now();
            printf("Neon %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            
            }
            break;
            
        case mix8_32:
            {
            std::vector<int8_t> Af(n*k);
            std::vector<float32_t> Bf(k*m);
            std::vector<float32_t> Df(n*m);

            matrix_init_rand(&Af[0], n*k);
            matrix_init_rand(&Bf[0], k*m);

            start_2 = std::chrono::high_resolution_clock::now();
            matrix_multiply_neon(&Af[0], &Bf[0], &Df[0], n, m, k);
            stop_2 = std::chrono::high_resolution_clock::now();
            printf("Neon [mix8_32] size: %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            }
            break;
            
        case check_mix8_16:
            {
            std::vector<int8_t> Af(n*k);
            std::vector<int8_t> Bf(k*m);
            std::vector<int16_t> Df(n*m);

            matrix_init_rand(&Af[0], n*k);
            matrix_init_rand(&Bf[0], k*m);

            start_2 = std::chrono::high_resolution_clock::now();
            matrix_multiply_neon(&Af[0], &Bf[0], &Df[0], n, m, k);
            stop_2 = std::chrono::high_resolution_clock::now();
            printf("Neon [mix8_32] size: %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            printf("===============================\n");

            std::vector<int16_t> Cf(n*m);
            matrix_init(&Cf[0], n, m, 0);
            std::vector<int16_t> Ef(n*m);
            start = std::chrono::high_resolution_clock::now();
            matrix_multiply_c(&Af[0], &Bf[0], &Ef[0], n, m, k);
            stop = std::chrono::high_resolution_clock::now();
            printf("C (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
            c_eq_neon = matrix_comp(&Ef[0], &Df[0], n, m);
            printf("Neon equal to C? %d\n", c_eq_neon);
            printf("===============================\n");
            }
            break;
            
        case check_mix8_32:
            {
            std::vector<int8_t> Af(n*k);
            std::vector<float32_t> Bf(k*m);
            std::vector<float32_t> Df(n*m);

            matrix_init_rand(&Af[0], n*k);
            matrix_init_rand(&Bf[0], k*m);

            start_2 = std::chrono::high_resolution_clock::now();
            matrix_multiply_neon(&Af[0], &Bf[0], &Df[0], n, m, k);
            stop_2 = std::chrono::high_resolution_clock::now();
            printf("Neon %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            printf("===============================\n");

            std::vector<float32_t> Cf(n*m);
            matrix_init(&Cf[0], n, m, 0);
            std::vector<float32_t> Ef(n*m);
            start = std::chrono::high_resolution_clock::now();
            matrix_multiply_c(&Af[0], &Bf[0], &Ef[0], n, m, k);
            stop = std::chrono::high_resolution_clock::now();
            printf("C (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
            c_eq_neon = matrix_comp(&Ef[0], &Df[0], n, m);
            printf("Neon equal to C? %d\n", c_eq_neon);
            printf("===============================\n");
            }
            break;
            
        case check_int16:
            {
            std::vector<int16_t> A(n*k);
            std::vector<int16_t> B(k*m);
            std::vector<int16_t> D(n*m);
                
            matrix_init_rand(&A[0], n*k);
            matrix_init_rand(&B[0], k*m);

            start_2 = std::chrono::high_resolution_clock::now();
            matrix_multiply_neon(&A[0], &B[0], &D[0], n, m, k);
            stop_2 = std::chrono::high_resolution_clock::now();
            printf("Neon %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            printf("===============================\n");

            std::vector<int16_t> C(n*m);
            matrix_init(&C[0], n, m, 0);
            std::vector<int16_t> E(n*m);
            start = std::chrono::high_resolution_clock::now();
            matrix_multiply_c(&A[0], &B[0], &E[0], n, m, k);
            stop = std::chrono::high_resolution_clock::now();
            printf("C (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
            c_eq_neon = matrix_comp(&E[0], &D[0], n, m);
            printf("Neon equal to C? %d\n", c_eq_neon);
            printf("===============================\n");
            }
            break;
            
        case check_float32:
            {
            std::vector<float32_t> Af(n*k);
            std::vector<float32_t> Bf(k*m);
            std::vector<float32_t> Df(n*m);

            matrix_init_rand(&Af[0], n*k);
            matrix_init_rand(&Bf[0], k*m);


            start_2 = std::chrono::high_resolution_clock::now();
            matrix_multiply_neon(&Af[0], &Bf[0], &Df[0], n, m, k);
            stop_2 = std::chrono::high_resolution_clock::now();
            printf("Neon %d (use : %li us) \n", SIZE, std::chrono::duration_cast<std::chrono::microseconds>(stop_2 - start_2));
            printf("===============================\n");

            std::vector<float32_t> Cf(n*m);
            matrix_init(&Cf[0], n, m, 0);
            std::vector<float32_t> Ef(n*m);
            start = std::chrono::high_resolution_clock::now();
            matrix_multiply_c(&Af[0], &Bf[0], &Ef[0], n, m, k);
            stop = std::chrono::high_resolution_clock::now();
            printf("C (use : %li us) \n", std::chrono::duration_cast<std::chrono::microseconds>(stop - start));
            c_eq_neon = matrix_comp(&Ef[0], &Df[0], n, m);
            printf("Neon equal to C? %d\n", c_eq_neon);
            printf("===============================\n");
            }
            break;
            
    }
}