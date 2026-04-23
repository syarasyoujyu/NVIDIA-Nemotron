# 何をするか
データの拡張

# 各ファイルの特徴

## gen_result.py
1.data/vocab.jsonlによってトークンを
デコーディング  
2.problems.jsonlから問題データを取得
3.generations.jsonl(モデルによる推論結果)をもとに問題とその出力結果をcsvにまとめる