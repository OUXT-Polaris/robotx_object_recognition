# ブイ認識

## environment
on macos
```
brew install opencv
```

## 構成
- 学習
    自前のデータセットで行う
    kerasでモデル構築

- 推論
    学習ずみの重みをtensorflowのC++ apiで処理する
    雑多な画像処理はopencvでやる

## TODO
- ブイでないものをどう判定するか
    異常認識の枠組み
