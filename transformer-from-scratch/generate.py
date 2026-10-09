import argparse
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
from Transformer import Transformer
from training_helpers import VOCAB_SIZE, batch_tensors, greedy_decode, generated_sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tokens', nargs='+', type=int, help='3–7 token IDs from 3 to 23')
    args = parser.parse_args()
    if not 3 <= len(args.tokens) <= 7 or any(token < 3 or token >= VOCAB_SIZE for token in args.tokens):
        parser.error('Use 3–7 token IDs in [3, 23], as in the training task.')
    checkpoint = torch.load(ROOT / 'checkpoints/copy_model.pt', map_location='cpu', weights_only=True)
    model = Transformer(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    source, _, _ = batch_tensors([args.tokens])
    tokens, emitted_eos = generated_sequence(greedy_decode(model, source)[0])
    print('Input:    ', args.tokens)
    print('Generated:', tokens)
    print('EOS:      ', emitted_eos)


if __name__ == '__main__':
    main()
