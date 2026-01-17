# 最小LLMプロジェクト

教育目的で作成された最小限の言語モデル（LLM）の実装例です。

## 📝 概要

このプロジェクトには以下が含まれています：
- シンプルなトランスフォーマーベースの言語モデル
- 日本語学習データ
- モデルの訓練スクリプト
- テキスト生成機能

## 🗂️ プロジェクト構成

```
example002/
├── README.md              # このファイル
├── instruction.md         # 詳細な説明
├── requirements.txt       # 必要なパッケージ
├── model.py              # モデルの定義
├── train.py              # 訓練スクリプト
├── generate.py           # テキスト生成スクリプト
└── train_data.txt        # 学習データ
```

## 🚀 クイックスタート

### 1. 仮想環境の作成と有効化

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

### 2. 依存パッケージのインストール

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. モデルの訓練

```bash
python train.py
```

訓練には5-10分程度かかります（CPUの場合）。

### 4. テキスト生成

```bash
python generate.py
```

### 5. 仮想環境の無効化

```bash
deactivate
```

## 🤖 モデルの仕様

- **アーキテクチャ**: トランスフォーマーデコーダー
- **埋め込み次元**: 128
- **アテンションヘッド数**: 4
- **レイヤー数**: 2
- **コンテキスト長**: 64トークン
- **総パラメータ数**: 約50,000-100,000
- **訓練時間**: CPU上で数分程度

## 📊 使用例

```python
from model import TinyLLM
import torch

# モデルの読み込み
checkpoint = torch.load('tiny_llm.pth')
model = TinyLLM(vocab_size=checkpoint['vocab_size'])
model.load_state_dict(checkpoint['model_state_dict'])

# テキスト生成
context = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(context, max_new_tokens=200)
```

## ⚙️ カスタマイズ

`model.py` のパラメータを調整してモデルサイズを変更できます：

```python
model = TinyLLM(
    vocab_size=vocab_size,
    n_embd=128,      # 埋め込み次元
    n_head=4,        # アテンションヘッド数
    n_layer=2,       # レイヤー数
    block_size=64    # コンテキスト長
)
```

## 🔧 トラブルシューティング

### PyTorchのインストールに失敗する場合

```bash
# CPU版をインストール
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### メモリ不足エラーが発生する場合

`train.py` のパラメータを調整：
- `batch_size = 16`（デフォルト: 32）
- `block_size = 32`（デフォルト: 64）

## 📚 詳細なドキュメント

より詳細な情報は [instruction.md](instruction.md) を参照してください。

## ⚠️ 注意事項

これは教育目的の最小限の実装です。実用的なアプリケーションには以下の改善が必要です：
- より大きなモデルサイズ
- より多くの訓練データ
- より長い訓練時間
- GPU の使用
- 高度な訓練テクニック

## 📄 ライセンス

このコードは教育目的で自由に使用できます。

## 🤝 貢献

改善提案やバグ報告は Issue や Pull Request でお願いします。
