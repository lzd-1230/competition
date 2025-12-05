#  模型优化说明 (Qwen2-VL& janus_pro)

## QWen部分

### Conv3d优化

profiling过后发现一个Conv3D的性能问题，可以使用img2col的方案从而避免使用这个算子。

其中要注意权重转换。

```
class PatchEmbedLinear(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)
        self._linear_weight = None

    def _prepare_linear_weight(self):
        """
        将 Conv3D 权重转换为 Linear 格式并缓存
        保持与原始权重完全相同的数值精度
        """
        if self._linear_weight is None:
            # Conv3D 权重: (out_channels, in_channels, kt, kh, kw)
            conv_weight = self.proj.weight
            
            # 直接 reshape，不进行任何数值转换
            # 这确保了权重数值的完全一致性
            self._linear_weight = conv_weight.reshape(conv_weight.shape[0], -1)
            
            # 确保缓存权重的 dtype 与原始权重一致
            self._linear_weight = self._linear_weight.to(dtype=conv_weight.dtype)
            
        return self._linear_weight

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        使用 Linear 计算方式（矩阵乘法）替代 Conv3D
        保持与原始 Conv3D 相同的精度
        """
        # 保持与原始 Conv3D 完全相同的 dtype 处理
        target_dtype = self.proj.weight.dtype
        
        # 保持与 Conv3D 相同的 view 操作
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, 
            self.patch_size, self.patch_size
        )
        
        # 转换为目标 dtype（与原始实现一致）
        hidden_states = hidden_states.to(dtype=target_dtype)
        
        # 展平为 (batch, patch_dim)
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(batch_size, -1)
        
        # 准备 Linear 权重（首次调用时转换并缓存）
        linear_weight = self._prepare_linear_weight()
        
        # 直接使用目标 dtype 进行计算，避免不必要的类型转换
        # 这确保了与原始 Conv3D 完全相同的数值精度
        output = ops.matmul(hidden_states, linear_weight.T)
        
        # 保持与原始实现相同的输出格式
        return output.view(-1, self.embed_dim)
```

### minds pore.ops相关算子替换

- attn部分flash-attn直接换出现精度报错，因此换成`F.scaled_dot_product_attention`官方实现的普通attn

  ```
  class Attention(nn.Module):
      fused_attn: Final[bool]
  
      def __init__(
          self,
          dim: int,
          num_heads: int = 8,
          qkv_bias: bool = False,
          qk_norm: bool = False,
          attn_drop: float = 0.0,
          proj_drop: float = 0.0,
          norm_layer: nn.Module = nn.LayerNorm,
      ) -> None:
          super().__init__()
          assert dim % num_heads == 0, "dim should be divisible by num_heads"
          self.num_heads = num_heads
          self.head_dim = dim // num_heads
          self.scale = self.head_dim**-0.5
          # self.fused_attn = use_fused_attn()
          self.fused_attn = True
  
          self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
          self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
          self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
          self.attn_drop = nn.Dropout(attn_drop)
          self.proj = nn.Linear(dim, dim)
          self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()
  
      def forward(self, x: mindspore.Tensor) -> mindspore.Tensor:
          B, N, C = x.shape
          qkv = (
              self.qkv(x)
              .reshape(B, N, 3, self.num_heads, self.head_dim)
              .permute(2, 0, 3, 1, 4)
          )
          q, k, v = qkv.unbind(0)
          q, k = self.q_norm(q), self.k_norm(k)
  
          x = F.scaled_dot_product_attention(
              q,
              k,
              v,
              attn_mask=None,
              is_causal=False,
              dropout_p=0.0,  # 推理时dropout固定为0.0
          )
  
  ```

  

- `apply_multimodal_rotary_pos_emb` 中替换融合算子

  ```
  def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
      mrope_section = mrope_section * 2
      cos = ops.cat([m[i % 3] for i, m in enumerate(ops.split(cos, mrope_section, dim=-1))], dim=-1).unsqueeze(
          unsqueeze_dim
      )
      sin = ops.cat([m[i % 3] for i, m in enumerate(ops.split(sin, mrope_section, dim=-1))], dim=-1).unsqueeze(
          unsqueeze_dim
      )
  
      # q_embed = (q * cos) + (rotate_half(q) * sin)
      # k_embed = (k * cos) + (rotate_half(k) * sin)
      # 更换成融合算子
      from mindspore.ops import rotary_position_embedding
      q_embed = rotary_position_embedding(q, cos, sin, 0)
      k_embed = rotary_position_embedding(k, cos, sin, 0)
      return q_embed, k_embed
  ```

- RMSnorm算子更换成`F.rms_norm`

### 缓存prefill阶段的视觉token

**类似于PD分离的效果，将视觉部分在prefill阶段和decode阶段的重复计算消除**

可以消除的部分

### 特殊场景优化

针对`batch=1`的场景，部分操作如mask构建以及shape变换等可以进行一些小优化

比如attn这边可以优化的换轴操作

```
# 优化：batch=1推理场景，使用permute代替swapaxes，减少操作次数
# RoPE 后统一交换轴到 (B,H,*,D)
query_states = query_states.permute(0, 2, 1, 3)  # (B,T,H,D) -> (B,H,T,D)
key_states = key_states.permute(0, 2, 1, 3)  # (B,T,H_kv,D) -> (B,H_kv,T,D)
value_states = value_states.permute(0, 2, 1, 3)  # (B,T,H_kv,D) -> (B,H_kv,T,D)

```

## Janus 部分

与上面均可以做相同的优化，如ROPE和RMSNorm的算子替换和prefill阶段图片token的缓存

### conv

conv这边采用相同的优化，出现精度轻微对不上导致结果少个空格的问题，比较类似替换flash-attn出现的问题

## 评测结果

| 评测指标        | 平均得分    |
| --------------- | ----------- |
| 峰值显存得分    | 116.6667    |
| Prefill时延得分 | 351.853     |
| Decode时延得分  | 149.8906    |
| **总分**        | **206.137** |