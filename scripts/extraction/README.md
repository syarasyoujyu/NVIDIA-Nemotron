# 何をしているか
ルールベースでどこまで推論が合うかをパターンごとに分析  
詳細なルールはpatterns/~~に存在

# 各ファイルに関して

## split_train_by_pattern
データをパターンごとに分割  
大分類(bit_manipulation等)/中分類（deduce,guess）といったdir構成で分割＆保存
## generate_pattern_rule_reports
各パターンにおいて、patterns/で定義したルールでどこまで合うかをvallidation＆validation結果を保存