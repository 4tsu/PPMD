# ======================================================
## Pythonで書いた粒子系シミュレーションコードのmain
### 仮想的な並列計算環境をシミュレートし、その上でMDを走らせる
# ======================================================

import envs
import particle
import sim_box
import sdd

# ----------------------------

Sys_test = envs.System()