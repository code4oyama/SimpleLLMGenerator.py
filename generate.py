"""
訓練済みモデルを使ってテキストを生成するスクリプト
"""

import torch
from model import TinyLLM


class Tokenizer:
    """シンプルな文字レベルのトークナイザー"""
    
    def __init__(self, char_to_idx, idx_to_char):
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = len(char_to_idx)
    
    def encode(self, text):
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])


def generate_text(prompt='', max_new_tokens=300, temperature=0.8):
    """テキストを生成"""
    
    # デバイスの設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # モデルとトークナイザーを読み込む
    print('モデルを読み込んでいます...')
    checkpoint = torch.load('tiny_llm.pth', map_location=device)
    
    # トークナイザーの復元
    tokenizer = Tokenizer(
        checkpoint['char_to_idx'],
        checkpoint['idx_to_char']
    )
    
    # モデルの復元
    model = TinyLLM(
        vocab_size=checkpoint['vocab_size'],
        n_embd=128,
        n_head=4,
        n_layer=2,
        block_size=checkpoint['block_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f'使用デバイス: {device}')
    print(f'Temperature: {temperature}')
    print('=' * 60)
    
    # プロンプトをエンコード
    if prompt:
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        print(f'プロンプト: {prompt}')
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print('プロンプト: なし（ランダム生成）')
    
    print('=' * 60)
    print('生成されたテキスト:')
    print('=' * 60)
    
    # テキスト生成
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature)
    
    # デコードして表示
    text = tokenizer.decode(generated[0].tolist())
    print(text)
    print('=' * 60)


if __name__ == '__main__':
    # 異なる設定でいくつか生成
    print('例1: プロンプトなしで生成')
    generate_text(prompt='', max_new_tokens=200, temperature=0.8)
    
    print('\n\n例2: プロンプト付きで生成')
    generate_text(prompt='人工知能', max_new_tokens=200, temperature=0.8)
    
    print('\n\n例3: より確定的な生成（低temperature）')
    generate_text(prompt='機械学習', max_new_tokens=200, temperature=0.5)
