# Teach and Repeat

固定場景、同一起點，原速重播遙操實際送出的六軸 ServoJ 和夾爪命令，保留停頓與抓放時機。
直接沿用已跑通的 `teleop_config.py`，不需要另填型號、韌體、校準識別或 Leader JSON。
回放不需要 Leader 或相機，不重新計算映射、平滑或變速。

## 本機既有遙操

本 workspace 的研究 checkout 已接入共用錄製器；原本的遙操參數與夾爪映射保持原樣：

```bash
cd /home/dsa/project/dobot/x-trainer
python3 run_teleop.py --teach-dir teach_episodes
```

Enter 啟動後，靜止約 0.3 秒按 **R** 開始，完成示教並停穩後按 **E** 保存。
R/E 在控制週期邊界生效。若仍在動或回授過期，終端會顯示拒絕原因。
等「已保存」及檔名出現後，可繼續遙操回到起點；Ctrl+C 退出，再復位物件。
同一程序可以連續錄多段，E 之後的動作不會納入已保存的一段。

```bash
python3 run_repeat.py teach_episodes/teach_實際檔名.jsonl --execute --gripper-enable-torque
```

此本機入口會自動使用旁邊的 `teleop_config.py`。起點檢查通過後輸入 **REPLAY**，原速播放一次。
`--gripper-enable-torque` 明確允許在回放前啟用夾爪扭矩；若扭矩已開啟可省略。
回放結束不自動鬆爪或關扭矩。手臂必須已在 RobotMode 5；不會自動使能、清錯或歸位。
不必先另跑離線檢查；實機入口自己會在連線前驗證整個錄製檔。

## 公開 component 入口

安裝 Python 3.10+、`pyserial` 即可使用獨立 runtime；Leader/夾爪 SDK 在 `source/leisaac`。
公開 checkout 保留占位端點，使用 `--settings` 指向你既有且可信的 Python 設定檔：

```bash
cd /home/dsa/project/dobot/components/x-trainer
PYTHONPATH=source/leisaac python3 run_teleop.py \
  --execute --settings /path/to/teleop_config.py \
  --gripper --gripper-enable-torque --clutch-key b --teach-dir teach_episodes
PYTHONPATH=source/leisaac python3 run_repeat.py teach_episodes/teach_實際檔名.jsonl \
  --execute --settings /path/to/teleop_config.py --gripper-enable-torque
```

`b` 應使用現場已確認的 clutch 選項。未提供 `--settings` 時，優先讀取同目錄的
`teleop_config.local.py`，否則使用 `teleop_config.py`。本機設定不可提交。
設定檔是會執行的 Python，僅使用自己可信的既有檔案；它應只宣告設定，不連硬體。

## 檢查與限制

預設回放入口只驗證和預演，不開 socket、serial 或 camera：

```bash
python3 run_repeat.py teach_episodes/teach_實際檔名.jsonl
```

預演顯示完整命令總數、時長、起點和前十筆事件，可用 `--preview-events N` 調整顯示數量。
錄製檔使用 schema 2，只接受本版新錄製、具有完整結尾與連續序號的檔案；不執行舊診斷 trace。
隊列滿、磁碟錯誤、命令失敗或錄製中退出都使該段無效。錄製資料與報告不覆寫，也不可提交。

速度、夾爪原始位置／速度／加速度和已配置的關節限制沿用既有遙操設定。
手臂速度檢查以每筆錄製 ServoJ 的 `t`（point time）核對目標變化；命令的
`send_ns` 只用於重播時序，不會把網路傳輸間隔誤當成手臂運動時間。
若既有 `joint_limits_deg=None`，維持未指定，程式不推測型號或虛構機械限制。
回放前比對夾爪命令設定與錄製起點；開合端點允許兩種方向，按端點最小／最大值檢查。
實際運動不保證零誤差，請保持同一起點、物件、工具、負載與場景。

軟體中止閾值集中在 `teach_config.py` 的 `RepeatConfig.from_teleop`，可在**同一份**既有
`teleop_config.py` 中用可選的 `REPEAT_CHECKS` dictionary 覆寫個別值。預設：

| 檢查 | 預設值 |
| --- | --- |
| 起點／實際終點關節誤差 | 每軸 2 度 |
| 關節跟隨誤差 | 每軸 10 度 |
| 起點／實際終點夾爪誤差 | 30 原始計數 |
| 夾爪命令跟隨誤差 | 既有開合端點距離；允許抓物時未達空夾端點 |
| 手臂回授期限 | 既有 `stale_timeout_s` |
| 夾爪回授期限 | 0.3 秒 |
| 靜止窗口／關節變動／夾爪變動 | 0.3 秒／0.2 度／5 計數，且 moving=false |
| 等待時監控週期 | 0.01 秒 |
| 最大發送延遲 | 0.01 秒 |
| 起點與結尾穩定等待期限 | 5 秒 |

這些是可調的程式中止門檻，並非實機校準結果或安全認證。原速執行不會套用限速器修改內容；
超過目前配置限制直接拒絕。命令按單調時鐘絕對時刻發送，延遲超標或上一筆未完成而下一筆已到期就中止，
不跳幀、突發補發、重新連線或恢復。起點檢查會在 REPLAY 確認後再次執行。

回授過期、模式異常、跟隨超差、通訊失敗或 Ctrl+C 後不再發新軌跡命令，嘗試 `Stop()`。
正常完成也保留完整尾段並確認實際終點穩定，再 Stop。報告存於 `repeat_reports/`，包含來源 SHA256、
命令數、實際發送／回應時間與延遲、跟隨誤差、相對示教實際軌跡偏差、停止原因及 Stop 結果。
`commands_sent` 表示已取得回應；未取得回應的命令可能已到控制器，另以 attempted/unacknowledged 記錄。

## 離線驗證

```bash
python3 -m pip install -r requirements-teach-dev.txt
PYTHONPATH=source/leisaac:source/leisaac/leisaac/xtrainer_utils python3 -m pytest -q
python3 -m mypy --follow-imports=silent --ignore-missing-imports --check-untyped-defs \
  teach_config.py teach_episode.py teach_controls.py repeat_runtime.py repeat_hardware.py \
  run_repeat.py run_teleop.py nova_client.py nova_gripper.py teleop_config.py gripper_diagnostics.py
python3 -m ruff check teach_config.py teach_episode.py teach_controls.py repeat_runtime.py repeat_hardware.py \
  run_repeat.py run_teleop.py test_teach_repeat.py test_repeat_interfaces.py tests/test_teach_settings.py
```

測試使用假時鐘、記憶體 transport、假遙操及虛擬終端，涵蓋 R/E、實際夾爪寫入、時序與失敗停止。
實機再現尚未驗證；短示教的移動、停頓、抓取與放開及實際偏差需在當次授權下驗收。
