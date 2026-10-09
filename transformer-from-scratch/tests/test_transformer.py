import copy
import inspect
import pytest
import torch
from Transformer import Transformer
from MultiHeadAttention import MultiHeadAttention
from InputEmbedding import InputEmbedding
from masks import causal_mask, padding_mask
from training_helpers import batch_tensors, make_examples, PAD, copy_logits, configure_padding

torch.set_num_threads(2)


def small_model(dropout=0.):
    torch.manual_seed(7)
    return Transformer(src_vocab=24, tgt_vocab=24, D=16, heads=4,
                       d_ff=32, N=2, max_len=16, dropout=dropout)


def test_different_source_target_lengths_and_tensor_shape():
    model = small_model().eval()
    output = model(torch.tensor([[3, 4, 5, 2, 0], [6, 7, 2, 0, 0]]),
                   torch.tensor([[1, 3, 4], [1, 6, 7]]))
    assert output.shape == (2, 3, 24)
    assert torch.isfinite(output).all()


def test_explicit_causal_mask_blocks_future_token_changes():
    model = small_model().eval()
    source = torch.tensor([[3, 4, 5, 2]])
    target = torch.tensor([[1, 3, 4, 5]])
    changed = target.clone(); changed[0, -1] = 12
    torch.testing.assert_close(copy_logits(model, source, target)[:, :3],
                               copy_logits(model, source, changed)[:, :3])


def test_attention_matches_independent_scaled_dot_product_reference():
    torch.manual_seed(4)
    attention = MultiHeadAttention(D=8, heads=2)
    query, key, value = torch.randn(2, 3, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    def split(projected):
        return projected.reshape(2, -1, 2, 4).transpose(1, 2)
    q, k, v = split(attention.Q(query)), split(attention.K(key)), split(attention.V(value))
    scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / 2.
    expected = torch.einsum('bhqk,bhkd->bhqd', scores.softmax(-1), v)
    expected = attention.W_o(expected.transpose(1, 2).reshape(2, 3, 8))
    torch.testing.assert_close(attention(query, key, value), expected)


@pytest.mark.parametrize('rank', [2, 3, 4])
def test_masks_are_reusable_without_mutating_shape_or_values(rank):
    attention = MultiHeadAttention(D=8, heads=2)
    tensor = torch.randn(2, 3, 8)
    mask = causal_mask(3)
    if rank == 2:
        mask = mask[0, 0]
    elif rank == 3:
        mask = mask[:, 0].expand(2, -1, -1)
    before = mask.clone(); shape = mask.shape
    first = attention(tensor, tensor, tensor, mask)
    second = attention(tensor, tensor, tensor, mask)
    assert mask.shape == shape
    torch.testing.assert_close(mask, before)
    torch.testing.assert_close(first, second)


def test_padding_keys_have_no_effect_on_output():
    attention = MultiHeadAttention(D=8, heads=2)
    query, keys, values = torch.randn(1, 3, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    mask = torch.tensor([[[[True, True, False, False]]]])
    changed_keys, changed_values = keys.clone(), values.clone()
    changed_keys[:, 2:] = 1000; changed_values[:, 2:] = -1000
    torch.testing.assert_close(attention(query, keys, values, mask),
                               attention(query, changed_keys, changed_values, mask))


def test_padding_length_does_not_change_real_token_logits():
    model = small_model().eval()
    target = torch.tensor([[1, 3, 4]])
    compact = copy_logits(model, torch.tensor([[3, 4, 2]]), target)
    padded = copy_logits(model, torch.tensor([[3, 4, 2, 0, 0]]), target)
    torch.testing.assert_close(compact, padded, rtol=1e-5, atol=1e-6)


def test_completely_masked_rows_have_finite_forward_and_backward():
    attention = MultiHeadAttention(D=8, heads=2)
    tensor = torch.randn(1, 3, 8, requires_grad=True)
    output = attention(tensor, tensor, tensor, torch.zeros(3, 3, dtype=torch.bool))
    output.sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(tensor.grad).all()


def test_model_has_finite_gradients_through_encoder_decoder_and_ffn():
    model = small_model()
    source, target, labels = batch_tensors([(3, 4, 5), (6, 7, 8, 9)])
    loss = torch.nn.functional.cross_entropy(copy_logits(model, source, target).reshape(-1, 24), labels.reshape(-1), ignore_index=PAD)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_state_dict_round_trip_and_eval_determinism():
    model = small_model(.2).eval()
    clone = copy.deepcopy(model)
    clone.load_state_dict(model.state_dict())
    source, target, _ = batch_tensors([(3, 4, 5)])
    torch.testing.assert_close(model(source, target), clone(source, target))
    torch.testing.assert_close(model(source, target), model(source, target))
    assert 'src_input.pe' in model.state_dict()


def test_embedding_dropout_and_buffer_device():
    embedding = InputEmbedding(24, 9, 12, dropout=.5)
    tokens = torch.tensor([[3, 4, 5]])
    embedding.eval()
    torch.testing.assert_close(embedding(tokens), embedding(tokens))
    embedding.train()
    assert not torch.equal(embedding(tokens), embedding(tokens))
    assert embedding.pe.device == embedding.embedding.weight.device
    assert embedding(tokens).shape == (1, 3, 9)


@pytest.mark.parametrize('D,heads', [(8, 3), (8, 0), (0, 2)])
def test_invalid_head_count_is_rejected(D, heads):
    with pytest.raises(ValueError):
        MultiHeadAttention(D, heads)


def test_invalid_masks_and_sequence_length_are_rejected():
    attention = MultiHeadAttention(8, 2)
    tensor = torch.randn(1, 3, 8)
    with pytest.raises(TypeError):
        attention(tensor, tensor, tensor, torch.ones(3, 3))
    with pytest.raises(ValueError):
        attention(tensor, tensor, tensor, torch.ones(7, 7, dtype=torch.bool))
    with pytest.raises(ValueError):
        InputEmbedding(24, 8, 2, .1)(torch.tensor([[1, 3, 4]]))


def test_dataset_splits_are_disjoint_and_shift_is_correct():
    train = make_examples(30, 42)
    validation = make_examples(10, 43, excluded=train)
    assert set(train).isdisjoint(validation)
    _, target, labels = batch_tensors([(3, 4, 5)])
    assert target.tolist() == [[1, 3, 4, 5]]
    assert labels.tolist() == [[3, 4, 5, 2]]


def test_tiny_batch_can_be_overfit():
    torch.manual_seed(42)
    model = Transformer(24, 24, 16, 4, 32, 1, 16, 0.)
    configure_padding(model)
    source, target, labels = batch_tensors([(3, 4, 5), (6, 7, 8)])
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    initial = None
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(copy_logits(model, source, target).reshape(-1, 24), labels.reshape(-1))
        if initial is None:
            initial = loss.item()
        loss.backward(); optimizer.step()
    assert loss.item() < initial * .1
    assert (copy_logits(model, source, target).argmax(-1) == labels).all()


def test_public_class_interfaces():
    from Encoder import Encoder
    from Decoder import Decoder
    from Encoder_Block import EncoderBlock
    from Decoder_Block import DecoderBlock
    from AddNorm import AddNorm
    from FeedForward import FeedForward
    constructors = {
        Transformer: ['self', 'src_vocab', 'tgt_vocab', 'D', 'heads', 'd_ff', 'N', 'max_len', 'dropout'],
        InputEmbedding: ['self', 'V', 'D', 'max_len', 'dropout'],
        MultiHeadAttention: ['self', 'D', 'heads'],
        Encoder: ['self', 'D', 'heads', 'd_ff', 'N'],
        Decoder: ['self', 'D', 'heads', 'd_ff', 'N'],
        EncoderBlock: ['self', 'D', 'heads', 'd_ff'],
        DecoderBlock: ['self', 'D', 'heads', 'd_ff'],
        FeedForward: ['self', 'D', 'd_ff'],
        AddNorm: ['self', 'D'],
    }
    for cls, names in constructors.items():
        assert list(inspect.signature(cls.__init__).parameters) == names
    signature = inspect.signature(Transformer.forward)
    assert list(signature.parameters) == ['self', 'src', 'tgt', 'src_mask', 'tgt_mask']
    assert signature.parameters['src_mask'].default is None
    assert signature.parameters['tgt_mask'].default is None


def test_encoder_and_decoder_sublayer_order():
    model = small_model().eval()
    events = []
    handles = []
    for prefix, block, names in [
        ('encoder', model.encoder.encoder[0], ['attention', 'norm1', 'feed_forward', 'norm2']),
        ('decoder', model.decoder.decoder[0],
         ['attention', 'norm1', 'cross_attention', 'norm2', 'feed_forward', 'norm3']),
    ]:
        for name in names:
            label = prefix + '.' + name
            handles.append(getattr(block, name).register_forward_hook(
                lambda module, args, output, label=label: events.append(label)))
    try:
        source, target, _ = batch_tensors([(3, 4, 5)])
        copy_logits(model, source, target)
    finally:
        for handle in handles:
            handle.remove()
    assert events == ['encoder.attention', 'encoder.norm1', 'encoder.feed_forward', 'encoder.norm2',
                      'decoder.attention', 'decoder.norm1', 'decoder.cross_attention', 'decoder.norm2',
                      'decoder.feed_forward', 'decoder.norm3']


def test_mask_policy_is_controlled_by_caller():
    model = small_model().eval()
    source = torch.tensor([[3, 4, 5, 2]])
    target = torch.tensor([[1, 3, 4, 5]])
    changed = target.clone(); changed[0, -1] = 12

    assert not torch.allclose(model(source, target)[:, :3], model(source, changed)[:, :3])
    torch.testing.assert_close(model(source, target, tgt_mask=causal_mask(4))[:, :3],
                               model(source, changed, tgt_mask=causal_mask(4))[:, :3])


def test_padding_configuration_is_explicit_and_preserves_other_embeddings():
    model = small_model()
    embedding = model.src_input.embedding
    assert embedding.padding_idx is None
    embedding(torch.tensor([PAD])).sum().backward()
    assert embedding.weight.grad[PAD].abs().sum() > 0
    model.zero_grad(set_to_none=True)
    before = embedding.weight.detach().clone()
    configure_padding(model)
    assert embedding.padding_idx == PAD
    torch.testing.assert_close(embedding.weight[1:], before[1:])
    assert torch.equal(embedding.weight[PAD], torch.zeros_like(embedding.weight[PAD]))
    embedding(torch.tensor([PAD, 3])).sum().backward()
    assert embedding.weight.grad[PAD].abs().sum() == 0
    assert embedding.weight.grad[3].abs().sum() > 0
