"""
最小限のLLM（言語モデル）の実装
トランスフォーマーベースの小さなモデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """自己注意機構"""
    
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        # クエリ、キー、バリューの線形変換
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        
        # 因果的マスク（未来のトークンを見ないように）
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
    
    def forward(self, x):
        B, T, C = x.size()  # バッチサイズ、シーケンス長、埋め込み次元
        
        # QKVを計算
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # マルチヘッドに分割
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # 注意スコアの計算
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # 注意を適用
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 出力の投影
        y = self.proj(y)
        return y


class FeedForward(nn.Module):
    """フィードフォワードネットワーク"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """トランスフォーマーブロック"""
    
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.attention = SelfAttention(n_embd, n_head, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # 残差接続付きの注意機構
        x = x + self.attention(self.ln1(x))
        # 残差接続付きのフィードフォワード
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyLLM(nn.Module):
    """最小限の言語モデル"""
    
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=2, block_size=64):
        super().__init__()
        
        self.block_size = block_size
        
        # トークン埋め込みと位置埋め込み
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        # トランスフォーマーブロック
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size) 
            for _ in range(n_layer)
        ])
        
        # 最終レイヤー
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, f"シーケンス長 {T} がブロックサイズ {self.block_size} を超えています"
        
        # 埋め込み
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # トランスフォーマーブロックを通す
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # 損失の計算（訓練時）
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """テキスト生成"""
        for _ in range(max_new_tokens):
            # コンテキストをブロックサイズに切り詰める
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # 予測を取得
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # サンプリング
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 新しいトークンを追加
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
