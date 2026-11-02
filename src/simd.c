/*
 * Marvin - an UCI/XBoard compatible chess engine
 * Copyright (C) 2015 Martin Danielsson
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <assert.h>
#include <stdlib.h>
#include <stdalign.h>
#ifdef USE_SSE
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#endif
#ifdef USE_AVX2
#include <immintrin.h>
#endif
#ifdef USE_NEON
#include <arm_neon.h>
#endif

#include "simd.h"
#include "utils.h"
#include "types.h"

#if defined(USE_AVX2)
#define NUM_REGS 16
#elif defined(USE_NEON)
#define NUM_REGS 8
#elif defined(USE_SSE)
#define NUM_REGS 8
#endif

#if defined(USE_AVX2)

static int32_t hsum_8x32(__m256i v)
{
    v = _mm256_add_epi32(v, _mm256_srli_si256(v, 8));
    v = _mm256_add_epi32(v, _mm256_srli_si256(v, 4));
    return _mm256_extract_epi32(v, 0) + _mm256_extract_epi32(v, 4);
}

int32_t simd_fully_connected(int16_t *inputs, int16_t *weights)
{
    int k;
    int niterations = NNUE_HIDDEN_LAYER_SIZE/NUM_REGS;

    __m256i vmin = _mm256_set1_epi16(0);
    __m256i vmax = _mm256_set1_epi16(NNUE_QUANT_QA);

    __m256i vsum = _mm256_setzero_si256();

    __m256i *pi = (__m256i*)inputs;
    __m256i *pw = (__m256i*)weights;

    for (k=0;k<niterations;k++,pi++,pw++) {
        __m256i vi = _mm256_load_si256(pi);
        __m256i vw = _mm256_load_si256(pw);

        __m256i vclamp = _mm256_max_epi16(vi, vmin);
        vclamp = _mm256_min_epi16(vclamp, vmax);

        __m256i vprod = _mm256_mullo_epi16(vclamp, vw);
        __m256i vres =  _mm256_madd_epi16(vprod, vclamp);

        vsum = _mm256_add_epi32(vsum, vres);
    }

    return hsum_8x32(vsum);
}

void simd_add(int16_t *inputs, int16_t *outputs)
{
    int k;
    int niterations = NNUE_HIDDEN_LAYER_SIZE/NUM_REGS;

    __m256i *pi = (__m256i*)inputs;
    __m256i *po = (__m256i*)outputs;

    for (k=0;k<niterations;k++) {
        po[k] = _mm256_add_epi16(po[k], pi[k]);
    }
}

void simd_sub(int16_t *inputs, int16_t *outputs)
{
    int k;
    int niterations = NNUE_HIDDEN_LAYER_SIZE/NUM_REGS;

    __m256i *pi = (__m256i*)inputs;
    __m256i *po = (__m256i*)outputs;

    for (k=0;k<niterations;k++) {
        po[k] = _mm256_sub_epi16(po[k], pi[k]);
    }
}

void simd_add_sub(int16_t* input, int16_t *output, int16_t *add, int16_t *sub)
{
    int k;
    int niterations = NNUE_HIDDEN_LAYER_SIZE/NUM_REGS;

    __m256i *pi = (__m256i*)input;
    __m256i *po = (__m256i*)output;
    __m256i *padd = (__m256i*)add;
    __m256i *psub = (__m256i*)sub;

    for (k=0;k<niterations;k++) {
        __m256i v = _mm256_add_epi16(pi[k], padd[k]);
        po[k] = _mm256_sub_epi16(v, psub[k]);
    }
}

void simd_add_sub2(int16_t* input, int16_t *output, int16_t *add, int16_t *sub1,
                   int16_t *sub2)
{
    int k;
    int niterations = NNUE_HIDDEN_LAYER_SIZE/NUM_REGS;

    __m256i *pi = (__m256i*)input;
    __m256i *po = (__m256i*)output;
    __m256i *padd = (__m256i*)add;
    __m256i *psub1 = (__m256i*)sub1;
    __m256i *psub2 = (__m256i*)sub2;

    for (k=0;k<niterations;k++) {
        __m256i v = _mm256_add_epi16(pi[k], padd[k]);
        __m256i v2 = _mm256_sub_epi16(v, psub1[k]);
        po[k] = _mm256_sub_epi16(v2, psub2[k]);
    }
}

void simd_add2_sub2(int16_t* input, int16_t *output, int16_t *add1,
                    int16_t *add2, int16_t *sub1, int16_t *sub2)
{
    int k;
    int niterations = NNUE_HIDDEN_LAYER_SIZE/NUM_REGS;

    __m256i *pi = (__m256i*)input;
    __m256i *po = (__m256i*)output;
    __m256i *padd1 = (__m256i*)add1;
    __m256i *padd2 = (__m256i*)add2;
    __m256i *psub1 = (__m256i*)sub1;
    __m256i *psub2 = (__m256i*)sub2;

    for (k=0;k<niterations;k++) {
        __m256i v = _mm256_add_epi16(pi[k], padd1[k]);
        __m256i v2 = _mm256_add_epi16(v, padd2[k]);
        __m256i v3 = _mm256_sub_epi16(v2, psub1[k]);
        po[k] = _mm256_sub_epi16(v3, psub2[k]);
    }
}

#elif defined(USE_NEON)

/*
 * AArch64 NEON implementation.
 *
 * The hidden layer contains 512 int16 values. NEON processes eight values
 * per 128-bit vector. The fully-connected operation intentionally preserves
 * the original x86 sequence: clamp, 16-bit multiply (including low-word
 * wraparound), then pairwise signed 16-bit multiply-accumulate.
 */
static int32_t hsum_4x32(int32x4_t v)
{
    return vaddvq_s32(v);
}

int32_t simd_fully_connected(int16_t *inputs, int16_t *weights)
{
    int k;
    int32x4_t vsum = vdupq_n_s32(0);
    const int16x8_t vmin = vdupq_n_s16(0);
    const int16x8_t vmax = vdupq_n_s16(NNUE_QUANT_QA);

    for (k = 0; k < NNUE_HIDDEN_LAYER_SIZE; k += 8) {
        int16x8_t vi = vld1q_s16(&inputs[k]);
        int16x8_t vw = vld1q_s16(&weights[k]);
        int16x8_t vclamp = vmaxq_s16(vi, vmin);
        int16x8_t vprod;
        int32x4_t vlo;
        int32x4_t vhi;

        vclamp = vminq_s16(vclamp, vmax);
        vprod = vmulq_s16(vclamp, vw);

        vlo = vmull_s16(vget_low_s16(vprod), vget_low_s16(vclamp));
        vhi = vmull_s16(vget_high_s16(vprod), vget_high_s16(vclamp));
        vsum = vaddq_s32(vsum, vaddq_s32(vlo, vhi));
    }

    return hsum_4x32(vsum);
}

void simd_add(int16_t *inputs, int16_t *outputs)
{
    int k;
    for (k = 0; k < NNUE_HIDDEN_LAYER_SIZE; k += 8) {
        int16x8_t a = vld1q_s16(&outputs[k]);
        int16x8_t b = vld1q_s16(&inputs[k]);
        vst1q_s16(&outputs[k], vaddq_s16(a, b));
    }
}

void simd_sub(int16_t *inputs, int16_t *outputs)
{
    int k;
    for (k = 0; k < NNUE_HIDDEN_LAYER_SIZE; k += 8) {
        int16x8_t a = vld1q_s16(&outputs[k]);
        int16x8_t b = vld1q_s16(&inputs[k]);
        vst1q_s16(&outputs[k], vsubq_s16(a, b));
    }
}

void simd_add_sub(int16_t *input, int16_t *output, int16_t *add,
                  int16_t *sub)
{
    int k;
    for (k = 0; k < NNUE_HIDDEN_LAYER_SIZE; k += 8) {
        int16x8_t v = vaddq_s16(vld1q_s16(&input[k]),
                                vld1q_s16(&add[k]));
        v = vsubq_s16(v, vld1q_s16(&sub[k]));
        vst1q_s16(&output[k], v);
    }
}

void simd_add_sub2(int16_t *input, int16_t *output, int16_t *add,
                   int16_t *sub1, int16_t *sub2)
{
    int k;
    for (k = 0; k < NNUE_HIDDEN_LAYER_SIZE; k += 8) {
        int16x8_t v = vaddq_s16(vld1q_s16(&input[k]),
                                vld1q_s16(&add[k]));
        v = vsubq_s16(v, vld1q_s16(&sub1[k]));
        v = vsubq_s16(v, vld1q_s16(&sub2[k]));
        vst1q_s16(&output[k], v);
    }
}

void simd_add2_sub2(int16_t *input, int16_t *output, int16_t *add1,
                    int16_t *add2, int16_t *sub1, int16_t *sub2)
{
    int k;
    for (k = 0; k < NNUE_HIDDEN_LAYER_SIZE; k += 8) {
        int16x8_t v = vaddq_s16(vld1q_s16(&input[k]),
                                vld1q_s16(&add1[k]));
        v = vaddq_s16(v, vld1q_s16(&add2[k]));
        v = vsubq_s16(v, vld1q_s16(&sub1[k]));
        v = vsubq_s16(v, vld1q_s16(&sub2[k]));
        vst1q_s16(&output[k], v);
    }
}

#elif defined(USE_SSE)

static int32_t hsum_4x32(__m128i v)
{
    v = _mm_add_epi32(v, _mm_srli_si128(v, 8));
    v = _mm_add_epi32(v, _mm_srli_si128(v, 4));
    return _mm_cvtsi128_si32(v);
}

int32_t simd_fully_connected(int16_t *inputs, int16_t *weights)
{
    int k;
    int niterations = NNUE_HIDDEN_LAYER_SIZE/NUM_REGS;

    __m128i vmin = _mm_set1_epi16(0);
    __m128i vmax = _mm_set1_epi16(NNUE_QUANT_QA);

    __m128i vsum = _mm_setzero_si128();

    __m128i *pi = (__m128i*)inputs;
    __m128i *pw = (__m128i*)weights;

    for (k=0;k<niterations;k++,pi++,pw++) {
        __m128i vi = _mm_load_si128(pi);
        __m128i vw = _mm_load_si128(pw);

        __m128i vclamp = _mm_max_epi16(vi, vmin);
        vclamp = _mm_min_epi16(vclamp, vmax);

        __m128i vprod = _mm_mullo_epi16(vclamp, vw);
        __m128i vres =  _mm_madd_epi16(vprod, vclamp);

        vsum = _mm_add_epi32(vsum, vres);
    }

    return hsum_4x32(vsum);
}

void simd_add(int16_t *inputs, int16_t *outputs)
{
    int k;
    int niterations = NNUE_HIDDEN_LAYER_SIZE/NUM_REGS;

    __m128i *pi = (__m128i*)inputs;
    __m128i *po = (__m128i*)outputs;

    for (k=0;k<niterations;k++) {
        po[k] = _mm_add_epi16(po[k], pi[k]);
    }
}

void simd_sub(int16_t *inputs, int16_t *outputs)
{
    int k;
    int niterations = NNUE_HIDDEN_LAYER_SIZE/NUM_REGS;

    __m128i *pi = (__m128i*)inputs;
    __m128i *po = (__m128i*)outputs;

    for (k=0;k<niterations;k++) {
        po[k] = _mm_sub_epi16(po[k], pi[k]);
    }
}

void simd_add_sub(int16_t* input, int16_t *output, int16_t *add, int16_t *sub)
{
    int k;
    int niterations = NNUE_HIDDEN_LAYER_SIZE/NUM_REGS;

    __m128i *pi = (__m128i*)input;
    __m128i *po = (__m128i*)output;
    __m128i *padd = (__m128i*)add;
    __m128i *psub = (__m128i*)sub;

    for (k=0;k<niterations;k++) {
        __m128i v = _mm_add_epi16(pi[k], padd[k]);
        po[k] = _mm_sub_epi16(v, psub[k]);
    }
}

void simd_add_sub2(int16_t* input, int16_t *output, int16_t *add, int16_t *sub1,
                   int16_t *sub2)
{
    int k;
    int niterations = NNUE_HIDDEN_LAYER_SIZE/NUM_REGS;

    __m128i *pi = (__m128i*)input;
    __m128i *po = (__m128i*)output;
    __m128i *padd = (__m128i*)add;
    __m128i *psub1 = (__m128i*)sub1;
    __m128i *psub2 = (__m128i*)sub2;

    for (k=0;k<niterations;k++) {
        __m128i v = _mm_add_epi16(pi[k], padd[k]);
        __m128i v2 = _mm_sub_epi16(v, psub1[k]);
        po[k] = _mm_sub_epi16(v2, psub2[k]);
    }
}

void simd_add2_sub2(int16_t* input, int16_t *output, int16_t *add1,
                    int16_t *add2, int16_t *sub1, int16_t *sub2)
{
    int k;
    int niterations = NNUE_HIDDEN_LAYER_SIZE/NUM_REGS;

    __m128i *pi = (__m128i*)input;
    __m128i *po = (__m128i*)output;
    __m128i *padd1 = (__m128i*)add1;
    __m128i *padd2 = (__m128i*)add2;
    __m128i *psub1 = (__m128i*)sub1;
    __m128i *psub2 = (__m128i*)sub2;

    for (k=0;k<niterations;k++) {
        __m128i v = _mm_add_epi16(pi[k], padd1[k]);
        __m128i v2 = _mm_add_epi16(v, padd2[k]);
        __m128i v3 = _mm_sub_epi16(v2, psub1[k]);
        po[k] = _mm_sub_epi16(v3, psub2[k]);
    }
}

#else

static int32_t screlu(int16_t input)
{
    int16_t val;

    val = CLAMP(input, 0, NNUE_QUANT_QA);
    return val*val;
}

int32_t simd_fully_connected(int16_t *inputs, int16_t *weights)
{
    int32_t output = 0;
    int     k;

    for (k=0;k<NNUE_HIDDEN_LAYER_SIZE;k++) {
        output += screlu(inputs[k])*weights[k];
    }

    return output;
}

void simd_add(int16_t *inputs, int16_t *outputs)
{
    int k;

    for (k=0;k<NNUE_HIDDEN_LAYER_SIZE;k++) {
        outputs[k] += inputs[k];
    }
}

void simd_sub(int16_t *inputs, int16_t *outputs)
{
    int k;

    for (k=0;k<NNUE_HIDDEN_LAYER_SIZE;k++) {
        outputs[k] -= inputs[k];
    }
}

void simd_add_sub(int16_t* input, int16_t *output, int16_t *add,
                  int16_t *sub)
{
    int k;

    for (k=0;k<NNUE_HIDDEN_LAYER_SIZE;k++) {
        output[k] = input[k] + add[k] - sub[k];
    }
}

void simd_add_sub2(int16_t* input, int16_t *output, int16_t *add,
                   int16_t *sub1, int16_t *sub2)
{
    int k;

    for (k=0;k<NNUE_HIDDEN_LAYER_SIZE;k++) {
        output[k] = input[k] + add[k] - sub1[k] - sub2[k];
    }
}

void simd_add2_sub2(int16_t* input, int16_t *output, int16_t *add1,
                    int16_t *add2, int16_t *sub1, int16_t *sub2)
{
    int k;

    for (k=0;k<NNUE_HIDDEN_LAYER_SIZE;k++) {
        output[k] = input[k] + add1[k] + add2[k] - sub1[k] - sub2[k];
    }
}
#endif
