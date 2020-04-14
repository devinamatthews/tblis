#include "../test.hpp"

#include "configs/include_configs.hpp"

#include "matrix/normal_matrix.hpp"

const config* const configs[] =
{
#define FOREACH_CONFIG(config) &config::instance(),
#include "configs/foreach_config.h"
};
constexpr auto num_configs = sizeof(configs)/sizeof(configs[0]);

/*
 * Assume:
 *  k_c <= 1024
 *  m_e/n_e <= 64
 *  alignment <= 64
 *  sizeof(T) <= 16
 *  general stride = 2
 */
char A_buf[64*1024*16] __attribute__((aligned(64)));
char B_buf[64*1024*16] __attribute__((aligned(64)));
char C_buf[64*64*16*2] __attribute__((aligned(64)));
char D_buf[64*64*16*2] __attribute__((aligned(64)));

TEMPLATED_TEST_CASE(gemm_ukr, T, all_types)
{
    for (auto i : range(num_configs))
    {
        auto& cfg = *configs[i];

        if (cfg.check() == -1) continue;

        len_type MR = cfg.gemm_mr.def<T>();
        len_type NR = cfg.gemm_nr.def<T>();
        len_type KR = cfg.gemm_kr.def<T>();
        len_type ME = cfg.gemm_mr.extent<T>();
        len_type NE = cfg.gemm_nr.extent<T>();
        len_type MC = cfg.gemm_mc.def<T>();
        len_type NC = cfg.gemm_nc.def<T>();
        len_type KC = cfg.gemm_kc.def<T>();
        len_type MX = cfg.gemm_mc.max<T>();
        len_type NX = cfg.gemm_nc.max<T>();
        len_type KX = cfg.gemm_kc.max<T>();

        INFO_OR_PRINT("ukernel: " << cfg.name);
        INFO_OR_PRINT("MR, NR, KR = " << MR << ", " << NR << ", " << KR);
        INFO_OR_PRINT("ME, NE     = " << ME << ", " << NE);
        INFO_OR_PRINT("MC, NC, KC = " << MC << ", " << NC << ", " << KC);
        INFO_OR_PRINT("MX, NX, KX = " << MX << ", " << NX << ", " << KX);
        INFO_OR_PRINT("row major? " << (cfg.gemm_row_major.value<T>() ? "yes" : "no"));
        INFO_OR_PRINT("flipped? " << (cfg.gemm_flip_ukr.value<T>() ? "yes" : "no"));

        std::array<std::pair<len_type,len_type>,10> mns
        {{
             {0, 0}, {1, 1}, {MR, 0}, {0, NR},
             {MR, 1}, {1, NR}, {MR, NR-1},
             {MR-1, NR}, {MR-1, NR-1}, {MR, NR}
        }};

        std::array<std::pair<stride_type,stride_type>,4> strides
        {{
             {NR, 1}, {1, MR}, {NR*2, 2}, {2, MR*2}
        }};

        std::array<len_type,8> ks = {0, 1, KC-1, KC, KC+1, KX-1, KX, KX+1};

        for (auto mn : mns)
        {
            auto m = mn.first;
            auto n = mn.second;

            for (auto k : ks)
            {
                INFO("m, n, k = " << m << ", " << n << ", " << k);

                for (auto stride : strides)
                {
                    auto rs_c = stride.first;
                    auto cs_c = stride.second;

                    INFO("rs_c, cs_c = " << rs_c << ", " << cs_c);

                    for (T beta : {0, 1, -1})
                    {
                        INFO("beta = " << beta);

                        matrix_view<T> A({m, k}, reinterpret_cast<T*>(A_buf), {1, ME});
                        matrix_view<T> B({k, n}, reinterpret_cast<T*>(B_buf), {NE, 1});
                        matrix_view<T> C({m, n}, reinterpret_cast<T*>(C_buf), {rs_c, cs_c});
                        matrix_view<T> D({m, n}, reinterpret_cast<T*>(D_buf), {rs_c, cs_c});

                        packed_matrix Ap(type_tag<T>::value, MR, k, A_buf, ME*k);
                        packed_matrix Bp(type_tag<T>::value, k, NR, B_buf, NE*k);
                        normal_matrix Cu(beta, false, m, n, C_buf, rs_c, cs_c);

                        for (len_type j = 0;j < ME*k;j++)
                            A.data()[j] = random_unit<T>();
                        for (len_type j = 0;j < NE*k;j++)
                            B.data()[j] = random_unit<T>();
                        for (len_type j = 0;j < MR*NR*2;j++)
                            C.data()[j] = D.data()[j] = random_unit<T>();

                        INFO("C before:\n" << C);

                        auxinfo_t aux{A_buf, B_buf, C_buf};
                        cfg.gemm_ukr.call<T>(m, n, k, A_buf, B_buf, &beta,
                                             C_buf, rs_c, cs_c, &aux);

                        INFO("C after:\n" << C);

                        gemm_ref<T>(1.0, A, B, beta, D);

                        INFO("C ref:\n" << D);

                        add(-1, C, 1, D);
                        T error = reduce<T>(REDUCE_NORM_2, D);

                        INFO("C error:\n" << D);

                        check("REF", error, m*n*k);
                    }
                }
            }
        }
    }
}
