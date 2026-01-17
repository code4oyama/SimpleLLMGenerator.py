"""
最小限のLLMを訓練するスクリプト
"""

import torch
import torch.nn as nn
from model import TinyLLM
import time


class Tokenizer:
    """シンプルな文字レベルのトークナイザー"""
    
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    def encode(self, text):
        return [self.char_to_idx[c] for c in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])


def load_data(file_path):
    """訓練データを読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def get_batch(data, block_size, batch_size):
    """ランダムなバッチを生成"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1]) for i in ix])
    return x, y


def train():
    # ハイパーパラメータ
    batch_size = 32
    block_size = 64
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 100
    
    # デバイスの設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用デバイス: {device}')
    
    # データの読み込み
    print('データを読み込んでいます...')
    text = load_data('train_data.txt')
    print(f'データサイズ: {len(text)} 文字')
    
    # トークナイザーの作成
    tokenizer = Tokenizer(text)
    print(f'語彙サイズ: {tokenizer.vocab_size}')
    
    # データのエンコード
    data = tokenizer.encode(text)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # モデルの作成
    print('モデルを初期化しています...')
    model = TinyLLM(
        vocab_size=tokenizer.vocab_size,
        n_embd=128,
        n_head=4,
        n_layer=2,
        block_size=block_size
    )
    model = model.to(device)
    
    # パラメータ数を表示
    n_params = sum(p.numel() for p in model.parameters())
    print(f'モデルのパラメータ数: {n_params:,}')
    
    # オプティマイザ
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 損失を評価する関数
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, split_data in [('train', train_data), ('val', val_data)]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split_data, block_size, batch_size)
                X, Y = X.to(device), Y.to(device)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    # 訓練ループ
    print('訓練を開始します...')
    start_time = time.time()
    
    for iter in range(max_iters):
        # 定期的に損失を評価
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            elapsed = time.time() - start_time
            print(f'ステップ {iter:4d} | 訓練損失: {losses["train"]:.4f} | 検証損失: {losses["val"]:.4f} | 時間: {elapsed:.1f}s')
        
        # バッチを取得
        xb, yb = get_batch(train_data, block_size, batch_size)
        xb, yb = xb.to(device), yb.to(device)
        
        # フォワード＆バックワード
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print('訓練が完了しました！')
    
    # モデルとトークナイザーを保存
    print('モデルを保存しています...')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'char_to_idx': tokenizer.char_to_idx,
        'idx_to_char': tokenizer.idx_to_char,
        'block_size': block_size,
    }, 'tiny_llm.pth')
    print('モデルを tiny_llm.pth に保存しました')
    
    # テスト生成
    print('\nテスト生成:')
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=200)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == '__main__':
    train()
