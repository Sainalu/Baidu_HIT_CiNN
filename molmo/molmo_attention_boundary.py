"""Test-only no-cache Molmo attention boundary; no model formula/parameter changes."""
import types
import paddle
from paddleformers.transformers.molmo import modeling


def rope_with_explicit_head_axis(
    q, k, cos, sin, rope_impl="interleave", unsqueeze_dim=1
):
    # Keep ordinary callers equivalent; compiled core receives already-expanded inputs.
    if cos.ndim == 3:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    rotate = (
        modeling.rotate_every_two if rope_impl == "interleave" else modeling.rotate_half
    )
    qt, kt = q.dtype, k.dtype
    q = q.astype(paddle.float32)
    k = k.astype(paddle.float32)
    return (q * cos + rotate(q) * sin).astype(qt), (k * cos + rotate(k) * sin).astype(
        kt
    )


def core(self, x, mask, cos, sin):
    return modeling.MolmoAttention.forward(
        self,
        hidden_states=x,
        attention_mask=mask,
        position_embeddings=(cos, sin),
        use_cache=False,
    )


def boundary(
    self,
    hidden_states,
    past_key_values=None,
    attention_mask=None,
    attn_mask_startend_row_indices=None,
    position_embeddings=None,
    use_cache=False,
):
    if (
        use_cache
        or past_key_values is not None
        or attn_mask_startend_row_indices is not None
    ):
        raise ValueError(
            "This CINN test boundary only covers no-cache dense-mask attention"
        )
    cos, sin = position_embeddings
    return self._cinn_test_core(
        hidden_states, attention_mask, cos.unsqueeze(1), sin.unsqueeze(1)
    )


def install(model, compile_core):
    modeling.apply_rotary_pos_emb = rope_with_explicit_head_axis
    for layer in model.model.layers:
        attn = layer.self_attn
        fn = types.MethodType(core, attn)
        attn._cinn_test_core = (
            paddle.jit.to_static(fn, full_graph=True, backend="CINN")
            if compile_core
            else fn
        )
        attn.forward = types.MethodType(boundary, attn)
