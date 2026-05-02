# 何をしているか
kaggleへの提出の自動化

# ファイルに関して
## upload_adapter_to_kaggle
Tinker に保存された LoRA アダプターのチェックポイントを Kaggle モデルにアップロードする。

1. `tinker checkpoint list` で最新の `sampler_weights/final` チェックポイントを取得する
2. チェックポイントアーカイブを Tinker からダウンロードし、Modal ボリューム (`/adapter/weights`) に展開する
3. 指定の Kaggle モデルインスタンス (`huikang/nemotron-adapter/Transformers/default`) が存在しなければ作成する
4. アダプターファイルを新バージョンとして Kaggle にアップロードする

実行には `.env` に `KAGGLE_API_TOKEN`, `TINKER_API_KEY`, `MODAL_TOKEN_ID`,
`MODAL_TOKEN_SECRET` が必要。

`make upload-adapter` は `.env` を読んで直接 Python スクリプトを実行するため、
事前に `modal setup` や `modal token set` を実行する必要はない。
