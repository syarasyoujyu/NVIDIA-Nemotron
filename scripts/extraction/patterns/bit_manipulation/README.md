# bit_manipulation

## このディレクトリの役割

- この README では、bit 系問題をどういう処理パターンとして見ているかをまとめる。
- 実際の一致判定コードは [validator.py](./validator.py) に置く。
- `generate_pattern_rule_reports.py --mode unmatched` は、この `validator.py` の仮説に収まらない行を `unmatched.csv` に落とす。

## extract で見ていること

- `scripts/extraction/extract_pattern/bit_prompt.py` 側では、規則を決め打ちせず、入出力の対応を中立に記録する。
- つまり report 側は「観測の整理」、exact 側は「処理パターンの仮説と検証」という役割分担にしている。

## 厳密に言えること

- 入力も出力も 8 ビットの二進文字列。
- 1 問ごとに例題が 7, 8, 9, 10 個ある。
- 問題文には候補として `bit shifts`, `rotations`, `XOR`, `AND`, `OR`, `NOT`, `majority`, `choice` が明示されている。
- つまり「各出力ビットが、入力ビットの位置変換とブール演算の合成で作られる」問題だと読むのが自然。
- ただし、学習データだけからは 1 つの正規形に一意決定できない。

## 強い仮説

- 多くの問題は「入力ビット位置をずらした列を 1 本, 2 本, 3 本組み合わせる」形で説明できる。
- 位置変換の基本部品は次の 3 系統と考えるのが妥当。
  - `ROT(k)` : 回転
  - `SHL(k)` : 左シフト
  - `SHR(k)` : 右シフト
- 出力ビットごとの局所規則としては、次の層に分けると扱いやすい。
  - 単項: `x_i`, `NOT x_i`, 定数 0/1
  - 二項: `AND`, `OR`, `XOR`, `AND-NOT`, `OR-NOT`, `XOR-NOT`
  - 三項以上: 2 本の変換結果を組み合わせた後、さらに別の変換結果と合成

## validator.py が今チェックしているパターン

### 1. unary / constant

- 各出力ビットが 1 つの入力ビット、またはその否定、または定数。
- 最も簡単で、位置対応だけ見ればよい。
- `validator.py` では `unary_or_constant_per_output_bit` として判定している。

### 2. binary stride-consistent

- 各出力ビットが「2 つの入力位置への同一演算」で表せる。
- しかも出力ビット 0,1,2,... に対して、参照位置が一定の stride で回っていく。
- この層は総当たりまたはビットペア列挙でかなり拾える。
- 今の validator では stride までは見ず、各出力ビットが局所 unary/binary 規則で説明できるかを見ている。
- 実装上の family 名は `local_unary_binary_boolean_rule`。

## まだ validator.py に入れていないが、次に足す候補

### 1. left/right stride + middle fill

- 左側はきれいな stride、右側もきれいな stride だが、中央だけ定数や別規則で埋まる。
- 提示された bit 解法メモの「左からの最長一致」「右からの最長一致」「中央補完」はこの層に対応している。

### 2. ternary local dependency

- 1 つの出力ビットが 3 つ以上の入力ビットに依存する層。
- 典型例は `SHL(x)` と `SHR(y)` の重なり方次第で、中央付近のビットが 3 入力依存になる場合。
- ここは二項演算ベースの solver だけでは落ちやすい。

## unmatched.csv をどう使うか

- まずは「各出力ビットが高々 2 入力に依存する」ケースを主対象にする。
- 式を直接列挙するより、出力ビットごとに「どの入力位置ペアがそのビットを作っているか」を列挙する方が探索空間を抑えやすい。
- その上で stride 一致を左端・右端から検出し、中央だけ別処理にするのが現実的。

- `unmatched.csv` に落ちた問題は、いまの `validator.py` の family では説明できなかったもの。
- そこを見ながら、stride 系や 3 入力依存系の family を 1 つずつ追加していく想定。

## 注意

- `XOR 差分` は観測量ではあるが、規則そのものを表してはいない。
- したがって bit の可視化では、`XOR` を主説明にしない方がよい。
