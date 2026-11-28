#if QSA_PACKED
// Sparse QSA for bf16 D=256, GQA12, on G17.
// Two bf16 terms retain enough softmax precision under relaxed NAX MMA.
// D and block IDs are contiguous; KV rows are aligned for uint4 loads.
using namespace mlx::steel;
// Two simdgroups split D=256 (warp*128) and share exchange[2][512]; one 32-key tile per KV pass.
static_assert(NSG == 2 && BK == 32, "sushi_qsa_nax_precise is written for 2 simdgroups and BK 32");
constexpr int BD = 256, LD = 264, TDH = 8, TK = BK / 16;
const int qL=q_shape[2], kL=k_shape[2], Hq=q_shape[1], Hk=k_shape[1];
const int gqa=Hq/Hk, KB=blocks_shape[2];
const int s=threadgroup_position_in_grid.x;
const int hk=threadgroup_position_in_grid.y;
const int bb=threadgroup_position_in_grid.z;
const ushort warp=simdgroup_index_in_threadgroup;
const ushort lane=thread_index_in_simdgroup;
const int tid=thread_index_in_threadgroup;
const int p=kL-qL+s, complete=(p+1)/RATIO;
const int sel_len=min(complete,KB)*RATIO, tail_start=complete*RATIO;
const int L=sel_len+p+1-tail_start;
const device int* blk=blocks+(long)bb*blocks_strides[0]+(long)s*blocks_strides[1];
#if QSA_PACKED
// k/v are the cache's packed affine words; ksc/kbi/vsc/vbi its per-group scales and biases.
const device uint32_t* Kp=k+bb*k_strides[0]+hk*k_strides[1];
const device uint32_t* Vp=v+bb*v_strides[0]+hk*v_strides[1];
const device T* Kscp=ksc+bb*ksc_strides[0]+hk*ksc_strides[1];
const device T* Kbip=kbi+bb*kbi_strides[0]+hk*kbi_strides[1];
const device T* Vscp=vsc+bb*vsc_strides[0]+hk*vsc_strides[1];
const device T* Vbip=vbi+bb*vbi_strides[0]+hk*vbi_strides[1];
#else
const device T* Kp=k+bb*k_strides[0]+hk*k_strides[1];
const device T* Vp=v+bb*v_strides[0]+hk*v_strides[1];
#endif
const device T* Qp=q+bb*q_strides[0]+(long)(hk*gqa)*q_strides[1]+(long)s*q_strides[2]+warp*128;
threadgroup T KV[BK*LD];
threadgroup float exchange[2][512];
using ST=NAXTile<float,1,TK>;
using OT=NAXTile<float,1,TDH>;
OT O; O.clear();
NAXTile<T,1,1> Q[TDH];
STEEL_PRAGMA_UNROLL
for (short d=0;d<TDH;++d) Q[d].load_rows(Qp+d*16, int(q_strides[1]), short(gqa));
const short2 coord=BaseNAXFrag::get_coord();
float2 max_score=float2(-3e38f),sum_score=float2(0.0f);
const float scale=scl[0]*1.44269504088896340736f;
for(int t0=0;t0<L;t0+=BK) {
  const int rows=min(BK,L-t0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int i=tid;i<BK*32;i+=64) {
    int r=i>>5,c=i&31; uint4 value=uint4(0);
    if(r<rows) {
      int pos=sushi_qsa_pos(blk,t0+r,sel_len,tail_start,RATIO);
#if QSA_PACKED
      value=sushi_qsa_unpack8<T>(Kp+(long)pos*k_strides[2],Kscp+(long)pos*ksc_strides[2],Kbip+(long)pos*kbi_strides[2],c,BITS,GS);
#else
      value=*((const device uint4*)(Kp+(long)pos*k_strides[2])+c);
#endif
    }
    *((threadgroup uint4*)(KV+r*LD)+c)=value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ST S; S.clear();
  STEEL_PRAGMA_UNROLL
  for(short ik=0;ik<TK;ik+=2) {
    STEEL_PRAGMA_UNROLL
    for(short d=0;d<TDH;++d) {
      NAXTile<T,2,1> K;
      K.template load<T,LD,1>(KV+ik*16*LD+warp*128+d*16);
      BaseNAXFrag::mma(S.frag_at(0,ik),S.frag_at(0,ik+1),Q[d].frag_at(0,0),
        metal::false_type{},K.frag_at(0,0),K.frag_at(1,0),metal::true_type{});
    }
    STEEL_PRAGMA_UNROLL
    for(short i=0;i<8;++i) {
      exchange[warp][lane*16+i]=S.frag_at(0,ik)[i];
      exchange[warp][lane*16+8+i]=S.frag_at(0,ik+1)[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for(short i=0;i<8;++i) {
      S.frag_at(0,ik)[i]+=exchange[1-warp][lane*16+i];
      S.frag_at(0,ik+1)[i]+=exchange[1-warp][lane*16+8+i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  STEEL_PRAGMA_UNROLL
  for(short ik=0;ik<TK;++ik) {
    STEEL_PRAGMA_UNROLL
    for(short i=0;i<8;++i) {
      S.frag_at(0,ik)[i]*=scale;
      if(ik*16+coord.x+(i%4)>=rows) S.frag_at(0,ik)[i]=-INFINITY;
    }
  }
  float2 new_max=max_score;
  S.template row_reduce<QsaMax>(new_max);
  S.template row_bin_op<QsaExpSub>(new_max);
  float2 factor=metal::exp2(max_score-new_max);
  max_score=new_max;
  sum_score*=factor;
  S.template row_reduce<QsaSum>(sum_score);
  O.template row_bin_op<QsaMul>(factor);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int i=tid;i<BK*32;i+=64) {
    int r=i>>5,c=i&31; uint4 value=uint4(0);
    if(r<rows) {
      int pos=sushi_qsa_pos(blk,t0+r,sel_len,tail_start,RATIO);
#if QSA_PACKED
      value=sushi_qsa_unpack8<T>(Vp+(long)pos*v_strides[2],Vscp+(long)pos*vsc_strides[2],Vbip+(long)pos*vbi_strides[2],c,BITS,GS);
#else
      value=*((const device uint4*)(Vp+(long)pos*v_strides[2])+c);
#endif
    }
    *((threadgroup uint4*)(KV+r*LD)+c)=value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  STEEL_PRAGMA_UNROLL
  for(short d=0;d<TDH;d+=2) {
    STEEL_PRAGMA_UNROLL
    for(short ik=0;ik<TK;++ik) {
      NAXTile<T,1,2> V;
      V.template load<T,LD,1>(KV+ik*16*LD+warp*128+d*16);
      NAXTile<T,1,1> Shi, Slo;
      for(short j=0;j<8;++j) {
        Shi.frag_at(0,0)[j]=T(S.frag_at(0,ik)[j]);
        float residual=S.frag_at(0,ik)[j]-float(Shi.frag_at(0,0)[j]);
        Slo.frag_at(0,0)[j]=T(residual);
      }
      BaseNAXFrag::mma(O.frag_at(0,d),O.frag_at(0,d+1),Shi.frag_at(0,0),
        metal::false_type{},V.frag_at(0,0),V.frag_at(0,1),metal::false_type{});
      BaseNAXFrag::mma(O.frag_at(0,d),O.frag_at(0,d+1),Slo.frag_at(0,0),
        metal::false_type{},V.frag_at(0,0),V.frag_at(0,1),metal::false_type{});
    }
  }
}
float2 inv=1.0f/sum_score;
O.template row_bin_op<QsaMul>(inv);
device T* Op=out+(((long)bb*Hq+hk*gqa)*qL+s)*BD+warp*128;
O.store_rows(Op,qL*BD,short(gqa));
#else
constexpr int D = 256;
constexpr int TDH = 8;
constexpr int SB = 8;
constexpr int PV_K = PV_TERMS == 2 ? 32 : 16;
threadgroup float xchg[2][8 * 32];

const int tq = int(threadgroup_position_in_grid.x);
const int kvh = int(threadgroup_position_in_grid.y);
const ushort dh = simdgroup_index_in_threadgroup;
const ushort lane = thread_index_in_simdgroup;
const short qid = lane >> 2;
const short fm = (qid & 4) | ((lane >> 1) & 3);
const short fn = ((qid & 2) | (lane & 1)) * 4;

const int qL = q_shape[2], kL = k_shape[2];
const int bb = int(threadgroup_position_in_grid.z);
const float scale2 = scl[0] * 1.44269504089f;
const int p = kL - qL + tq;
const int complete = (p + 1) >> 2;
const int nsel = min(TOPK, complete);
const int ntail = p + 1 - (complete << 2);
const int U = nsel + (ntail > 0 ? 1 : 0);
const int nsteps = (U + SB - 1) / SB;
const bool r1_ok = fm + 8 < GQA;
const int h0 = kvh * GQA + fm;
const int h1 = kvh * GQA + (r1_ok ? fm + 8 : fm);

const int64_t sqh = q_strides[1], sql = q_strides[2];
const uint skl = uint(k_strides[2]), svl = uint(v_strides[2]);
const short fcol = 2 * fn;
const device bfloat* kb = (const device bfloat*)k + bb * k_strides[0] + kvh * k_strides[1] + dh * 128 + fcol;
const device bfloat* vb = (const device bfloat*)v + bb * v_strides[0] + kvh * v_strides[1] + dh * 128 + fcol;
const device bfloat* qp0 = (const device bfloat*)q + bb * q_strides[0] + h0 * sqh + tq * sql + dh * 128 + fcol;
const device bfloat* qp1 = (const device bfloat*)q + bb * q_strides[0] + h1 * sqh + tq * sql + dh * 128 + fcol;
const device int* blk = blocks + bb * blocks_strides[0] + tq * blocks_strides[1];

constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, 32, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, PV_K, false, false, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> pv_op;
auto qa_ct = qk_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
auto kb_ct = qk_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
using qa_t = metal::remove_addrspace_t<decltype(qa_ct)>;
using kb_t = metal::remove_addrspace_t<decltype(kb_ct)>;
auto pa_ct = pv_op.template get_left_input_cooperative_tensor<PT, bfloat, float>();
auto vb_ct = pv_op.template get_right_input_cooperative_tensor<PT, bfloat, float>();
using pa_t = metal::remove_addrspace_t<decltype(pa_ct)>;
using vb_t = metal::remove_addrspace_t<decltype(vb_ct)>;
auto of0 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
auto of1 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
auto of2 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
auto of3 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
UNROLL for (short i = 0; i < 16; ++i) {
  of0[i] = 0.0f;
  of1[i] = 0.0f;
  of2[i] = 0.0f;
  of3[i] = 0.0f;
}
float max_s[2] = {-FLT_MAX, -FLT_MAX};
float sum_s[2] = {0.0f, 0.0f};

auto blk_at = [&](int u) -> int { return u < nsel ? blk[u] : complete; };
int nrow_b[4];
int ncol_n[2];
auto fetch = [&](int u0) {
  UNROLL for (short r = 0; r < 4; ++r) {
    const int u = u0 + ((16 * (r >> 1) + fm + 8 * (r & 1)) >> 2);
    nrow_b[r] = u < U ? blk_at(u) : 0;
  }
  UNROLL for (short f = 0; f < 2; ++f) {
    const int u = u0 + 4 * f + (fn >> 2);
    ncol_n[f] = u < nsel ? 4 : (u < U ? ntail : 0);
  }
};
fetch(0);

for (int step = 0; step < nsteps; ++step) {
  uint krow[4];
  UNROLL for (short r = 0; r < 4; ++r) {
    krow[r] = uint(min(nrow_b[r] * 4 + (fm & 3), kL - 1));
  }
  int nvis[2];
  UNROLL for (short f = 0; f < 2; ++f) {
    nvis[f] = ncol_n[f];
  }
  if (step + 1 < nsteps) {
    fetch((step + 1) * SB);
  }

  auto s_ct = qk_op.template get_destination_cooperative_tensor<qa_t, kb_t, float>();
  UNROLL for (short i = 0; i < 16; ++i) {
    s_ct[i] = 0.0f;
  }
  UNROLL for (short jj = 0; jj < TDH / 2; ++jj) {
    const vec<bfloat, 8> qa = *(const device vec<bfloat, 8>*)(qp0 + 32 * jj);
    const vec<bfloat, 8> qb = r1_ok ? *(const device vec<bfloat, 8>*)(qp1 + 32 * jj) : vec<bfloat, 8>(0);
    const vec<bfloat, 8> a0 = *(const device vec<bfloat, 8>*)(kb + (krow[0] * skl + 32 * jj));
    const vec<bfloat, 8> a1 = *(const device vec<bfloat, 8>*)(kb + (krow[1] * skl + 32 * jj));
    const vec<bfloat, 8> a2 = *(const device vec<bfloat, 8>*)(kb + (krow[2] * skl + 32 * jj));
    const vec<bfloat, 8> a3 = *(const device vec<bfloat, 8>*)(kb + (krow[3] * skl + 32 * jj));
    UNROLL for (short j = 0; j < 4; ++j) {
      qa_ct[j] = qa[j];
      qa_ct[4 + j] = qb[j];
      qa_ct[8 + j] = qa[4 + j];
      qa_ct[12 + j] = qb[4 + j];
      kb_ct[j] = a0[j];
      kb_ct[4 + j] = a1[j];
      kb_ct[8 + j] = a2[j];
      kb_ct[12 + j] = a3[j];
      kb_ct[16 + j] = a0[4 + j];
      kb_ct[20 + j] = a1[4 + j];
      kb_ct[24 + j] = a2[4 + j];
      kb_ct[28 + j] = a3[4 + j];
    }
    qk_op.run(qa_ct, kb_ct, s_ct);
  }
  vec<float, 8> s[2];
  UNROLL for (short i = 0; i < 8; ++i) {
    s[0][i] = s_ct[i];
    s[1][i] = s_ct[8 + i];
  }

  UNROLL for (short f = 0; f < 2; ++f) {
    threadgroup float* mine = xchg[dh];
    const threadgroup float* peer = xchg[1 - dh];
    UNROLL for (short i = 0; i < 8; ++i) {
      mine[lane * 8 + i] = s[f][i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    UNROLL for (short i = 0; i < 8; ++i) {
      s[f][i] += peer[lane * 8 + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  UNROLL for (short f = 0; f < 2; ++f) {
    UNROLL for (short j = 0; j < 4; ++j) {
      const bool ok = j < nvis[f];
      s[f][j] = ok ? s[f][j] * scale2 : -INFINITY;
      s[f][4 + j] = ok ? s[f][4 + j] * scale2 : -INFINITY;
    }
  }
  float factor[2];
  UNROLL for (short i = 0; i < 2; ++i) {
    float m = -INFINITY;
    UNROLL for (short f = 0; f < 2; ++f) {
      m = max(m, max(max(s[f][4 * i], s[f][4 * i + 1]), max(s[f][4 * i + 2], s[f][4 * i + 3])));
    }
    m = max(m, simd_shuffle_xor(m, ushort(1)));
    m = max(m, simd_shuffle_xor(m, ushort(8)));
    const float new_max = max(max_s[i], m);
    float rs = 0.0f;
    UNROLL for (short f = 0; f < 2; ++f) {
      UNROLL for (short j = 0; j < 4; ++j) {
        s[f][4 * i + j] = fast::exp2(s[f][4 * i + j] - new_max);
        rs += s[f][4 * i + j];
      }
    }
    rs += simd_shuffle_xor(rs, ushort(1));
    rs += simd_shuffle_xor(rs, ushort(8));
    factor[i] = fast::exp2(max_s[i] - new_max);
    max_s[i] = new_max;
    sum_s[i] = sum_s[i] * factor[i] + rs;
  }
  if (simd_any(factor[0] != 1.0f || factor[1] != 1.0f)) {
    UNROLL for (short h = 0; h < 2; ++h) {
      UNROLL for (short j = 0; j < 4; ++j) {
        of0[8 * h + j] *= factor[0];
        of0[8 * h + 4 + j] *= factor[1];
        of1[8 * h + j] *= factor[0];
        of1[8 * h + 4 + j] *= factor[1];
        of2[8 * h + j] *= factor[0];
        of2[8 * h + 4 + j] *= factor[1];
        of3[8 * h + j] *= factor[0];
        of3[8 * h + 4 + j] *= factor[1];
      }
    }
  }
  UNROLL for (short f = 0; f < 2; ++f) {
    vec<PT, 8> pc[PV_TERMS];
    {
      vec<float, 8> rest = s[f];
      UNROLL for (short tt = 0; tt < PV_TERMS; ++tt) {
        UNROLL for (short i = 0; i < 8; ++i) {
          const PT piece = PT(rest[i]);
          pc[tt][i] = piece;
          rest[i] -= float(piece);
        }
      }
    }
    UNROLL for (short id = 0; id < TDH; id += 2) {
      const vec<bfloat, 8> ra = *(const device vec<bfloat, 8>*)(vb + (krow[2 * f] * svl + 16 * id));
      const vec<bfloat, 8> rb = *(const device vec<bfloat, 8>*)(vb + (krow[2 * f + 1] * svl + 16 * id));
      UNROLL for (short j = 0; j < 4; ++j) {
        vb_ct[j] = ra[j];
        vb_ct[4 + j] = rb[j];
        vb_ct[8 + j] = ra[4 + j];
        vb_ct[12 + j] = rb[4 + j];
        if (PV_K == 32) {
          vb_ct[16 + j] = ra[j];
          vb_ct[20 + j] = rb[j];
          vb_ct[24 + j] = ra[4 + j];
          vb_ct[28 + j] = rb[4 + j];
        }
      }
      UNROLL for (short tt = 0; tt < PV_TERMS; tt += PV_K / 16) {
        UNROLL for (short i = 0; i < 8; ++i) {
          pa_ct[i] = pc[tt][i];
          if (PV_K == 32) {
            pa_ct[8 + i] = pc[tt + 1][i];
          }
        }
        if (id == 0) {
          pv_op.run(pa_ct, vb_ct, of0);
        } else if (id == 2) {
          pv_op.run(pa_ct, vb_ct, of1);
        } else if (id == 4) {
          pv_op.run(pa_ct, vb_ct, of2);
        } else {
          pv_op.run(pa_ct, vb_ct, of3);
        }
      }
    }
  }
}

UNROLL for (short i = 0; i < 2; ++i) {
  if (i == 1 && !r1_ok) {
    continue;
  }
  const float rr = 1.0f / sum_s[i];
  device bfloat* o = (device bfloat*)out + ((size_t(bb) * (2 * GQA) + (i == 0 ? h0 : h1)) * qL + tq) * D + dh * 128 + fcol;
  UNROLL for (short jj = 0; jj < TDH / 2; ++jj) {
    vec<bfloat, 8> w;
    UNROLL for (short j = 0; j < 4; ++j) {
      float e0, e1;
      if (jj == 0) {
        e0 = of0[4 * i + j];
        e1 = of0[8 + 4 * i + j];
      } else if (jj == 1) {
        e0 = of1[4 * i + j];
        e1 = of1[8 + 4 * i + j];
      } else if (jj == 2) {
        e0 = of2[4 * i + j];
        e1 = of2[8 + 4 * i + j];
      } else {
        e0 = of3[4 * i + j];
        e1 = of3[8 + 4 * i + j];
      }
      w[j] = bfloat(e0 * rr);
      w[4 + j] = bfloat(e1 * rr);
    }
    *(device vec<bfloat, 8>*)(o + 32 * jj) = w;
  }
}
#endif
