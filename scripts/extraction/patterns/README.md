# exact_patterns

このディレクトリは、各問題タイプごとに

- `README.md`: 何を観測し、どういう処理パターンとして整理しているか
- `validator.py`: その仮説パターンに実データが収まるかを判定するコード

を同じ場所に置くためのディレクトリです。

`scripts/extraction/extract_pattern/` が「レポート用に問題を読み解く場所」だとすると、  
`scripts/exact_patterns/` は「solver や unmatched 生成のために、仮説パターンを明示して検証する場所」です。

`generate_pattern_rule_reports.py --mode unmatched` を使うと、
各タイプの `validator.py` に書かれた仮説で説明できない問題を
`data/patterns/<pattern>/unmatched.csv` として保存できます。

```bash
python3 scripts/extraction/generate_pattern_rule_reports.py --mode unmatched
```

各タイプの詳細は以下です。

- [bit_manipulation/README.md](./bit_manipulation/README.md)
- [text_decryption/README.md](./text_decryption/README.md)
- [roman_numeral/README.md](./roman_numeral/README.md)
- [unit_conversion/README.md](./unit_conversion/README.md)
- [gravity_distance/README.md](./gravity_distance/README.md)
- [equation_transformation/README.md](./equation_transformation/README.md)
