# ブイ認識

## environment
on macos
```
# for cpp
brew install opencv
```

## trained models
https://drive.google.com/open?id=1jDuOb2zpX2N8MwJxrDY__jxIEnbZpGsz 

## 構成
- API
    - 
- 学習
    - 自前のデータセットで行う
    - kerasでモデル構築
    - `main_k.py`がkerasのメインコード

- 推論
    - 学習ずみの重みをtensorflowのC++ apiで処理する
    - 雑多な画像処理はopencvでやる
    - tensorflow用にポートする部分 https://qiita.com/cvusk/items/a2a0a11815de491cf3e5
    - Cで動かす(上の方が良さげ) https://qiita.com/yukiB/items/1ea109eceda59b26cd64#4-kerastensorflow%E3%81%A7%E4%BD%9C%E6%88%90%E3%81%97%E3%81%9F%E3%83%A2%E3%83%87%E3%83%AB%E3%82%92c%E3%81%8B%E3%82%89%E5%AE%9F%E8%A1%8C

## TODO
- ブイでないものをどう判定するか
    - 異常認識の枠組み
    - 背景画像を学習させて背景クラスを作る
    - convolutionが効いているかどうかの確認
- ROS化
    - `function(x): (画像x: 2d ndarray) -> (各クラスに所属する確率の分布: [float])`という関数を内側でコールするROSのpythonノード
        - ROIに従って切り出した後の画像をもらって、CNNの出力を返すようなノードという感じ

