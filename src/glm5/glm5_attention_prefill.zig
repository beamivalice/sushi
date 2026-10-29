//! Split-softmax latent attention partials shared by the serial and overlay kernels.
//! Needs `glm5_latent.header` for `SUSHI_LATENT`.
pub const common: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint lane=thread_position_in_threadgroup.x;
    \\const uint head=threadgroup_position_in_grid.y;
    \\const uint row=threadgroup_position_in_grid.z/uint(SPLITS);
    \\const uint part=threadgroup_position_in_grid.z%uint(SPLITS);
    \\constexpr uint ITEMS=(uint(D)+31u)/32u;
    \\float query[ITEMS],acc[ITEMS];
    \\for(uint j=0;j<ITEMS;++j) {uint d=lane+j*32u; query[j]=d<uint(D)?float(q[(row*uint(H)+head)*uint(D)+d]):0.0f;acc[j]=0.0f;}
    \\const uint pos=uint(offset)+row;
    \\const uint count=SELECTED?2051u:pos+1u;
    \\const uint chunk=(count+uint(SPLITS)-1u)/uint(SPLITS);
    \\const uint begin=part*chunk,end=min(count,begin+chunk);
    \\float maximum=-INFINITY,denom=0.0f;
    \\for(uint k=begin;k<end;++k) {
    \\  const int token=SELECTED?selected[row*2051u+k]:int(k);
    \\  if(token<0 || uint(token)>pos || uint(token)>=uint(length)) continue;
    \\  float values[ITEMS]; float dot=0.0f;
    \\  for(uint j=0;j<ITEMS;++j) {uint d=lane+j*32u;values[j]=d<uint(D)?float(SUSHI_LATENT(cache,uint(token),d,uint(D))):0.0f;dot+=query[j]*values[j];}
    \\  dot=simd_sum(dot)*float(scale);
    \\  float next=max(maximum,dot),old=precise::exp(maximum-next),p=precise::exp(dot-next);
    \\  denom=denom*old+p;
    \\  for(uint j=0;j<ITEMS;++j) acc[j]=acc[j]*old+p*values[j];
    \\  maximum=next;
    \\}
    \\const uint base=((row*uint(H)+head)*uint(SPLITS)+part);
;
pub const partial_tail: [:0]const u8 =
    \\for(uint j=0;j<ITEMS;++j) {uint d=lane+j*32u;if(d<uint(D)) partial[base*uint(D)+d]=acc[j];}
    \\if(lane==0) {stats[base*2u]=maximum;stats[base*2u+1u]=denom;}
;
