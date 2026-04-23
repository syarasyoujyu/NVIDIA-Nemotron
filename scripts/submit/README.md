# 何をしているか
kaggleへの提出の自動化

# ファイルに関して
## upload_adapter_to_kaggle
Tinker に保存された LoRA アダプターのチェックポイントを Kaggle モデルにアップロードする。

1. `tinker checkpoint list` で最新の `sampler_weights/final` チェックポイントを取得する
2. チェックポイントアーカイブを Tinker からダウンロードし、Modal ボリューム (`/adapter/weights`) に展開する
3. 指定の Kaggle モデルインスタンス (`huikang/nemotron-adapter/Transformers/default`) が存在しなければ作成する
4. アダプターファイルを新バージョンとして Kaggle にアップロードする

実行には `env.json` に `KAGGLE_API_TOKEN` と `TINKER_API_KEY` が必要。
