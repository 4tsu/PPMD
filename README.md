# PPMD
Pseudo Parallel Molecular Dynamics

仮想的にプロセス並列で動作するMDシミュレーション。

MPIとか使わずに、並列アルゴリズムの挙動を調べることができる。

ロードバランシングアルゴリズムの挙動を調べる目的で実装。

人に見せるつもりで書いないが、コードを使った結果を発表してしまったので、公開しておく。

## 内訳
`two`ディレクトリに2次元シミュレーションコードが、`three`に3次元のコードが入っている。
プログラムのmainは`run.py`で、import元を書き換えて2Dと3Dを切り替える。
`run.py`では、シミュレーションボックスの大きさ、粒子数、カットオフ距離、Bookkeeping法のマージン、ロードバランシングアルゴリズム、試行回数を指定できる。
また、既存の粒子配置の読み込みも可能。

まとめて何回か実行したいとき（計算速度の平均をとったり）は`run_hpc.py`を使用のこと。
