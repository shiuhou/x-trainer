# Teach and Repeat v1

固定場景、同一起點，按示教的原速重播六軸手臂和夾爪命令，包含停頓與抓放時機。
不需要 Leader、相機或訓練模型來回放。重播命令不等於零誤差的機械運動；實機誤差由回放報告量測。

本功能在公開 `components/x-trainer` 中實作。本機研究 checkout 不會自動獲得這些改動。
第一版只執行新錄製的 Teach episode，原有 `teleop_traces/*.jsonl` 保留為診斷資料，不能拿來執行。

## 安裝與本機設定

獨立 Python 環境只需 Python 3.10+、`pyserial`；相機可選，另外需要 OpenCV。
Isaac Sim、Leader SDK 與相機都不會在預設離線回放入口載入。

```bash
cd /home/dsa/project/dobot/components/x-trainer
python3 -m venv .venv-teach
.venv-teach/bin/pip install pyserial
cp teach_repeat.example.json repeat.local.json
cp teach_leader.example.json leader.local.json
```

範例的 `null`／`SET_...` 故意不可執行。請由現場已核對的設定填入，不能將測試 fixture 當成實機數值。
兩個 `*.local.json`、`teach_episodes/`、`repeat_reports/` 都不提交。使用其他輸出目錄時也必須保持私有。

`repeat.local.json` 的欄位：

| 群組 | 必填內容 |
| --- | --- |
| `identity` | `robot_id`、`model`、`firmware`、`calibration_id`；校準識別應對應已核對的工具、TCP、user frame、負載及夾爪安裝／開合方向 |
| `nova` | `host`、`dashboard_port`、`feedback_port`、`socket_timeout_s`、`stop_timeout_s` |
| `gripper` | `port`、`baudrate`、`servo_id`、`model_number`、`open_position`、`closed_position`、`command_speed`、`command_acceleration`、`serial_timeout_s` |
| `limits` | 六組 `joint_limits_deg` 下／上界，以及六軸 `max_speed_deg_s` |
| `checks` | 六軸起點／終點容許值 `joint_start_tolerance_deg`、六軸跟隨容許值 `joint_tracking_tolerance_deg`；其餘如下 |

`checks` 的其餘欄位均需明確填入：

- `gripper_start_tolerance`：起點和示教實際終點的夾爪位置容許值，以原始位置計數表示。
- `gripper_tracking_tolerance`：命令與實際夾爪位置的最大差值；抓住物件可能無法到達空夾端點，需按現場抓取條件設定。
- `feedback_timeout_s`／`gripper_feedback_timeout_s`：手臂／夾爪回授有效期限。
- `stationary_window_s`、`stationary_joint_tolerance_deg`、`stationary_gripper_tolerance`：一段時間內的最大關節／夾爪位置變化；夾爪 `moving` 也必須為 false。
- `monitor_period_s`：等待期間的監控間隔。
- `max_lateness_s`：發送相對於原始時刻的最大延遲。
- `completion_timeout_s`：起點靜止確認及回放後末端穩定確認的逾時。

`leader.local.json` 必須填入 `LeaderConfig` 的全部欄位，包括既有的 joint IDs、offsets、signs、trigger endpoints 與 sampling rate。
它只供遙操使用。不要重新猜測校準資料。

機器／firmware／校準識別由操作員核對後填入，程式比對錄製與回放的設定，**不會自動探測 Nova 型號、firmware、TCP 或負載**。
夾爪的 servo ID/model 會實際唯讀核對。開／閉兩端可遞增或遞減，範圍檢查使用兩端的最小／最大值。

## 錄製

沒有當前 session 的硬體操作授權時，只能跑下面的離線選項檢查：

```bash
python3 run_teleop.py --teach-dir teach_episodes --gripper
```

取得遙操授權並核對工作區、急停及設定後，才使用實機入口：

```bash
PYTHONPATH=source/leisaac python3 run_teleop.py \
  --execute --config repeat.local.json --leader-config leader.local.json \
  --gripper --gripper-enable-torque --clutch-key b --teach-dir teach_episodes
```

`b` 僅適用於當次確認過的按鈕；否則按現場設定選擇。此入口保持原有遙操映射、限速、clutch 與診斷 trace，
預設 profile 的速度上限若超過本機已審查上限，須明確提供六個 `--max-speed` 值。程式不會提高本機限制。
`--gripper-enable-torque` 是明確的遙操 opt-in；啟動確認後才啟用扭矩，trigger 移動達既有 deadband 才開始夾爪命令，
退出時恢復本次改變的扭矩狀態。回放入口完全不寫扭矩。

1. 在終端 Enter 確認後，先遙操準備好手臂／夾爪與物件位置。
2. 靜止後，在**終端**按 `R` 開始。R／E 不佔用 Leader 的 clutch 按鈕。
3. 按原本方式完成示教；原始停頓、clutch 期間的 hold 命令都保留。
4. 停穩後按 `E`。若回授過期、仍在動、尚未到達允許狀態，會顯示原因並拒絕該按鍵；停穩後再按。
5. 等待「已保存」及檔案路徑。之後可繼續遙操回起點，或再按 R 錄下一段；Ctrl+C 退出。

每段獨立命名，不覆寫。錄製中異常退出、丟資料、隊列滿、命令失敗或磁碟錯誤，該段都不可執行。
保存後的 episode 不會因為之後遙操退出而變成無效。錄製檔不含本機端點，但仍是私有原始實驗資料。

## 離線檢查與單次回放

預設命令會完整驗證檔案，列出時長、起點和每筆命令的相對時刻；不開任何硬體連線：

```bash
python3 run_repeat.py teach_episodes/teach_....jsonl
python3 run_repeat.py teach_episodes/teach_....jsonl --config repeat.local.json
```

先用既有流程手動回到錄製起點，退出遙操，復位物件。
夾爪扭矩必須用已授權的既有操作流程事先啟用，並保持在錄製初始位置。
取得當次回放授權後：

```bash
PYTHONPATH=source/leisaac python3 run_repeat.py teach_episodes/teach_....jsonl \
  --execute --config repeat.local.json
```

程式先唯讀核對 RobotMode 5、新鮮回授、起點與靜止狀態；輸入 `REPLAY` 後重新檢查，才播放一次。
不符合就拒絕，不自動歸位。回放不連 Leader／相機，不啟用、清錯、重新連線或恢復動作。

回放原樣使用錄製的 ServoJ 参数及夾爪 position/speed/acceleration，不重算 trigger、不濾波、不變速、不跳幀。
如果任何命令發送太晚，或上一筆尚未完成而下一筆已到期，就中止，不能密集補發。
夾爪唯讀輪詢和位置寫入共用 serial 鎖，讀取造成的等待也計入 deadline；系統不保證硬即時調度。

等待及命令回應後會檢查新鮮回授、模式、跟隨誤差；阻塞中的 I/O 受設定逾時約束，不能保證立即停止。
Ctrl+C、失敗和正常完成都嘗試 `Stop()`。TCP 故障可能使 Stop 失敗，報告會記錄；實體急停仍需可觸及。
回放結尾保持夾爪扭矩／位置，不自動放開物件。末端穩定確認不額外發 ServoJ。

每次實機入口會在 `repeat_reports/` 另存報告，含來源 SHA256、命令數、各命令發送／回應時刻和延遲、
跟隨誤差、相對示教實際關節軌跡的偏差、完成或中止原因、Stop 結果。
示教實際轨跡只在報告比較時按時間線性插值，不改變執行命令。
報告 `commands_sent` 是取得完整命令回應的次數；`commands_attempted` 與 `unacknowledged_event_seq`
指出嘗試但未取得回應的最後一筆，該命令可能已送到控制器。

## 離線驗證與實機驗收

```bash
python3 -m pip install -r requirements-teach-dev.txt
python3 -m pytest -q test_teach_repeat.py test_repeat_interfaces.py test_phase3.py test_episode_qa.py
# Full suite, including upstream FakeDynamixel tests:
PYTHONPATH=source/leisaac:source/leisaac/leisaac/xtrainer_utils python3 -m pytest -q
python3 -m mypy --follow-imports=silent --ignore-missing-imports --check-untyped-defs \
  teach_config.py teach_episode.py teach_controls.py repeat_runtime.py repeat_hardware.py \
  run_repeat.py run_teleop.py nova_client.py nova_gripper.py teleop_config.py gripper_diagnostics.py
python3 -m ruff check teach_config.py teach_episode.py teach_controls.py repeat_runtime.py repeat_hardware.py \
  run_repeat.py run_teleop.py nova_client.py nova_gripper.py teleop_config.py gripper_diagnostics.py \
  test_teach_repeat.py test_repeat_interfaces.py
```

離線測試使用假時鐘、記憶體 transport 和虛擬終端，並測試完整遙操入口的 R／E 錄製整合。
這些結果不是 firmware 相容性或實機安全證明。實機驗收尚待當次授權：
新錄一段移動／停頓／抓取／放開的短示教，手動復位後播放一次，核對時序、目標及實際偏差，
以現場設定的容許值判定結果。
