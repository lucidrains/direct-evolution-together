## Mummy WebSocket server and room evolution state engine for direct-evolution-together.

import std/[algorithm, atomics, hashes, json, locks, os, sets, strutils, sysrand, tables, times]
import mummy, mummy/routers
import nimpy
import zippy

const
  history_limit = 100
  max_username_len = 24
  max_message_len = 2000
  max_goal_name_len = 32
  max_task_name_len = 24
  max_room_name_len = 64
  max_cid_len = 40

  min_population_size = 3
  max_population_size = 10_000
  control_max_population_size = 10_000
  default_population_size = 30
  default_control_population_size = 30
  default_control_min_votes = 20
  default_crowd_shard_threshold = 16
  max_round_min_votes = 1024

  min_votes_per_user = 1
  max_votes_per_user = 256
  default_max_votes_per_user = 20

  min_mut_sigma = 0.05
  max_mut_sigma = 1.0
  default_mut_sigma = 0.30

  min_elite_frac = 0.0
  max_elite_frac = 0.50
  default_elite_frac = 0.20

  min_crossover_rate = 0.0
  max_crossover_rate = 1.0
  default_crossover_rate = 0.75

  min_brood_size = 1
  max_brood_size = 8
  default_brood_size = 1
  max_required_broods = 32

  default_min_votes = 20
  default_countdown_ms = 8000'i64
  vote_burst_window_ms = 1000'i64
  vote_burst_max_ops = 64
  max_lineage_nodes = 300
  max_lineage_cap = 4096
  lineage_cap_multiplier = 3
  default_hydrate_max_items = 16
  ticker_sleep_ms = 40
  default_ping_interval_ms = 20_000'i64
  default_stale_timeout_ms = 60_000'i64
  default_evolve_watchdog_ms = 30_000'i64
  # Compress JSON payloads above 512 bytes over WebSockets (history, state, lineage).
  gzip_min_bytes = 512

type
  EvoJobKind {.pure.} = enum
    init_room = "init_room"
    create = "create"
    evolve = "evolve"
    resize = "resize"

  GoalParams = object
    max_votes_per_user: int
    min_votes: int
    mut_sigma: float
    elite_frac: float
    crossover_rate: float
    brood_size: int

  RuntimeConfig = object
    min_votes: int
    min_votes_explicit: bool
    min_votes_override: bool
    countdown_ms: int64
    ping_interval_ms: int64
    stale_timeout_ms: int64
    evolve_watchdog_ms: int64
    population_size: int
    params: GoalParams

  TaskConfig = object
    is_control: bool
    sim_phase_enabled: bool
    default_auto_evolve: bool
    default_pop_size: int
    max_pop_size: int
    default_min_votes: int
    crowd_shard_threshold: int
    has_elite_override: bool
    default_elite_frac: float
    default_brood_size: int

  PendingClient = object
    room: string
    username: string
    cid: string
    upgrade_seq: uint64
    is_reconnect: bool
    initial_goal: string
    initial_task: string
    initial_pop_size: int
    initial_params: GoalParams
    initial_env_params_json: string
    gzip: bool
    can_evolve_now: bool
    lazy_goals: bool
    defer_adapters: bool
    defer_lineage: bool
    chat_limit: int
    active_goal: string

  ClientInfo = object
    ws: WebSocket
    room: string
    username: string
    cid: string
    upgrade_seq: uint64
    last_seen_ms: int64
    last_ping_ms: int64
    vote_timestamps_ms: seq[int64]
    gzip: bool
    can_evolve_now: bool
    lazy_goals: bool
    defer_adapters: bool
    defer_lineage: bool
    chat_limit: int
    active_goal: string

  PendingPersist = object
    room: string
    username: string
    message: string

  BroodFixedSlot = object
    slot: int
    genome_json: string

  BroodGroup = object
    slot: int
    parents: seq[int]
    op: string
    candidates_json: seq[string]
    ## Per-brood selection: WebSocket -> candidate index (0 ..< candidates_json.len).
    votes: Table[WebSocket, int]

  GoalPopulation = ref object
    name: string
    task: string
    generation: int
    genomes_json: string
    sharded_genomes_json: string
    genome_items: seq[string]
    uuid_to_idx: Table[string, int]
    head_uuid: string
    env_params_json: string
    lineage_json: string
    lean_lineage_json: string
    lineage_total: int
    ## Per-candidate vote map: WebSocket -> +1 (up) or -1 (down).
    votes: seq[Table[WebSocket, int]]
    phase: string
    sim_completed: int
    auto_evolve: bool
    countdown_ends_at_ms: int64
    evolving: bool
    evolving_started_at_ms: int64
    params: GoalParams
    brood_size_active: int
    brood_pop_size: int
    brood_top_score: int
    brood_fixed: seq[BroodFixedSlot]
    brood_groups: seq[BroodGroup]

  RoomEvolution = ref object
    goal_order: seq[string]
    goals: Table[string, GoalPopulation]
    pending_creates: HashSet[string]
    pending_create_tasks: Table[string, string]
    pending_create_params: Table[string, GoalParams]
    pending_create_env_params: Table[string, string]
    default_params: GoalParams
    initializing: bool

  EvoJob = object
    kind: EvoJobKind
    room: string
    goal: string
    task: string
    generation: int
    username: string
    genomes_json: string
    env_params_json: string
    scores: seq[int]
    up_counts: seq[int]
    population_size: int
    params: GoalParams

  PendingBroadcast = object
    targets: seq[WebSocket]
    payload: string

  VoteStats = tuple[up, down, score: int]

func default_goal_params(): GoalParams {.inline.} =
  GoalParams(
    max_votes_per_user: default_max_votes_per_user,
    min_votes: 0,
    mut_sigma: default_mut_sigma,
    elite_frac: default_elite_frac,
    crossover_rate: default_crossover_rate,
    brood_size: default_brood_size,
  )

func default_runtime_config(): RuntimeConfig {.inline.} =
  RuntimeConfig(
    min_votes: default_min_votes,
    min_votes_explicit: false,
    min_votes_override: false,
    countdown_ms: default_countdown_ms,
    ping_interval_ms: default_ping_interval_ms,
    stale_timeout_ms: default_stale_timeout_ms,
    evolve_watchdog_ms: default_evolve_watchdog_ms,
    population_size: default_population_size,
    params: default_goal_params(),
  )

# Bootstrap fallbacks only. Python `register_task_configs` installs the authoritative
# per-task policy (population, quorum, elite fraction) during server startup.
func default_task_configs(): Table[string, TaskConfig] =
  result = init_table[string, TaskConfig]()
  result["image"] = TaskConfig(
    is_control: false,
    default_pop_size: default_population_size,
    max_pop_size: max_population_size,
    default_min_votes: 0,
    crowd_shard_threshold: default_crowd_shard_threshold,
    has_elite_override: false,
    default_elite_frac: default_elite_frac,
    default_brood_size: default_brood_size,
  )
  result["lunarlander"] = TaskConfig(
    is_control: true,
    sim_phase_enabled: true,
    default_auto_evolve: false,
    default_pop_size: 16,
    max_pop_size: 128,
    default_min_votes: 4,
    crowd_shard_threshold: 64,
    has_elite_override: true,
    default_elite_frac: 0.25,
    default_brood_size: default_brood_size,
  )
  result["cartpole"] = TaskConfig(
    is_control: true,
    default_pop_size: default_control_population_size,
    max_pop_size: control_max_population_size,
    default_min_votes: default_control_min_votes,
    crowd_shard_threshold: default_crowd_shard_threshold,
    has_elite_override: true,
    default_elite_frac: 0.0,
    default_brood_size: default_brood_size,
  )
  result["halfcheetah"] = TaskConfig(
    is_control: true,
    default_pop_size: default_control_population_size,
    max_pop_size: control_max_population_size,
    default_min_votes: default_control_min_votes,
    crowd_shard_threshold: default_crowd_shard_threshold,
    has_elite_override: true,
    default_elite_frac: 0.05,
    default_brood_size: default_brood_size,
  )
  let populora_cfg = TaskConfig(
    is_control: true,
    sim_phase_enabled: true,
    default_auto_evolve: false,
    default_pop_size: 8,
    max_pop_size: control_max_population_size,
    default_min_votes: 4,
    crowd_shard_threshold: default_crowd_shard_threshold,
    has_elite_override: true,
    default_elite_frac: 0.25,
    default_brood_size: default_brood_size,
  )
  for tid in ["microduck", "unitree_go2", "so101"]:
    result[tid] = populora_cfg
  result["external"] = TaskConfig(
    is_control: true,
    default_pop_size: default_control_population_size,
    max_pop_size: max_population_size,
    default_min_votes: default_control_min_votes,
    crowd_shard_threshold: default_crowd_shard_threshold,
    has_elite_override: true,
    default_elite_frac: 0.10,
    default_brood_size: default_brood_size,
  )

var
  state_lock: Lock
  ready_lock: Lock
  ready_cond: Cond

  cfg = default_runtime_config()
  task_configs = default_task_configs()
  clients: Table[WebSocket, ClientInfo]
  pending_upgrades: Table[WebSocket, PendingClient]
  next_upgrade_seq: uint64 = 0
  rooms: Table[string, HashSet[WebSocket]]
  room_history: Table[string, seq[string]]
  room_evolution: Table[string, RoomEvolution]
  dirty_rooms: HashSet[string]
  pending_saves: seq[PendingPersist]
  pending_evo_jobs: seq[EvoJob]
  admin_password = ""
  evolve_token = ""

  server_ptr: Server = nil
  server_ready: Atomic[bool]
  server_failed: Atomic[bool]
  stop_ticker: Atomic[bool]
  server_thread: Thread[tuple[host: string, port: int]]
  ticker_thread: Thread[int]
  is_running = false

init_lock(state_lock)
init_lock(ready_lock)
init_cond(ready_cond)

template with_state_lock(body: untyped) =
  with_lock state_lock:
    {.gcsafe.}:
      body

proc quiet_logger(level: LogLevel; args: varargs[string]) {.gcsafe.} =
  if level == ErrorLevel:
    echo_logger(level, args)

proc now_ms(): int64 {.gcsafe, inline.} =
  let t = get_time()
  t.to_unix * 1000 + int64(t.nanosecond div 1_000_000)

proc utc_time_str(): string {.gcsafe.} =
  now().utc.format("HH:mm:ss")

proc constant_time_eq(a, b: string): bool {.gcsafe.} =
  if a.len != b.len:
    return false
  var diff = 0'u8
  for i in 0 ..< a.len:
    diff = diff or (uint8(a[i]) xor uint8(b[i]))
  diff == 0

func is_valid_room(room: string): bool =
  if room.len == 0 or room.len > max_room_name_len:
    return false
  for ch in room:
    if ch notin {'a'..'z', 'A'..'Z', '0'..'9', '_', '-'}:
      return false
  true

func is_auto_anon(name: string): bool {.inline.} =
  name.len == 9 and name.to_lower_ascii().starts_with("anon-") and
    name[5 .. ^1].all_chars_in_set(HexDigits)

func strip_collision_suffixes(raw: string): string =
  ## Strip trailing disambiguation counters (`-2`..`-99`) so collisions increment
  ## (`-2` -> `-3`) instead of stacking (`-2` -> `-2-2`).
  result = raw
  while not is_auto_anon(result):
    let dash = result.rfind('-')
    if dash <= 0:
      break
    let head = result[0 ..< dash]
    let tail = result[dash + 1 .. ^1]
    if head.to_lower_ascii() == "anon" or tail.len notin 1..2 or tail[0] == '0' or
       not tail.all_chars_in_set(Digits) or parse_int(tail) < 2:
      break
    result = head

proc sanitize_username(raw: string): string {.gcsafe.} =
  var cleaned = raw.strip()
  if cleaned.len > max_username_len:
    cleaned = cleaned[0 ..< max_username_len]
  if cleaned.len > 0:
    return cleaned
  let bytes = urandom(2)
  "anon-" & to_hex(bytes[0]).to_lower_ascii() & to_hex(bytes[1]).to_lower_ascii()

func sanitize_goal_name(raw: string): string =
  result = raw.strip()
  if result.len > max_goal_name_len:
    result = result[0 ..< max_goal_name_len]

func sanitize_cid(raw: string): string =
  let trimmed = raw.strip()
  if trimmed.len == 0:
    return ""
  result = new_string_of_cap(min(trimmed.len, max_cid_len))
  for ch in trimmed:
    if ch in {'a'..'z', 'A'..'Z', '0'..'9', '_', '-'}:
      result.add(ch)
      if result.len >= max_cid_len:
        break

proc sanitize_task_name_locked(raw: string): string {.gcsafe.} =
  let cleaned = raw.strip().to_lower_ascii()
  {.gcsafe.}:
    if cleaned in task_configs:
      return cleaned
  "image"

proc task_cfg_locked(task: string): TaskConfig {.gcsafe.} =
  let clean = sanitize_task_name_locked(task)
  {.gcsafe.}:
    if clean in task_configs:
      return task_configs[clean]
  TaskConfig(
    is_control: false,
    default_pop_size: default_population_size,
    max_pop_size: max_population_size,
    default_min_votes: 0,
    crowd_shard_threshold: default_crowd_shard_threshold,
    has_elite_override: false,
    default_elite_frac: default_elite_frac,
    default_brood_size: default_brood_size,
  )

proc max_pop_for_task_locked(task: string): int {.gcsafe, inline.} =
  task_cfg_locked(task).max_pop_size

proc default_pop_for_task_locked(task: string; base_pop: int): int {.gcsafe, inline.} =
  let tc = task_cfg_locked(task)
  if tc.is_control: min(base_pop, tc.default_pop_size) else: base_pop

proc default_min_votes_for_task_locked(task: string): int {.gcsafe, inline.} =
  task_cfg_locked(task).default_min_votes

proc default_elite_frac_for_task_locked(task: string; fallback: float): float {.gcsafe, inline.} =
  let tc = task_cfg_locked(task)
  if tc.has_elite_override: tc.default_elite_frac else: fallback

proc default_brood_size_for_task_locked(task: string; fallback: int): int {.gcsafe, inline.} =
  let tc = task_cfg_locked(task)
  if tc.default_brood_size > 1: clamp(tc.default_brood_size, min_brood_size, max_brood_size)
  else: clamp(fallback, min_brood_size, max_brood_size)

func clamp_goal_params(p: GoalParams): GoalParams {.inline.} =
  GoalParams(
    max_votes_per_user: clamp(p.max_votes_per_user, min_votes_per_user, max_votes_per_user),
    min_votes: (if p.min_votes > 0: clamp(p.min_votes, 1, max_round_min_votes) else: 0),
    mut_sigma: clamp(p.mut_sigma, min_mut_sigma, max_mut_sigma),
    elite_frac: clamp(p.elite_frac, min_elite_frac, max_elite_frac),
    crossover_rate: clamp(p.crossover_rate, min_crossover_rate, max_crossover_rate),
    brood_size: clamp(if p.brood_size > 0: p.brood_size else: default_brood_size, min_brood_size, max_brood_size),
  )

proc clamp_control_min_votes_locked(task: string; p: var GoalParams) {.gcsafe, inline.} =
  # Control tasks should be clearable by one person's vote budget.
  if task_cfg_locked(task).is_control and p.min_votes > p.max_votes_per_user:
    p.min_votes = max(min_votes_per_user, p.max_votes_per_user)

proc parse_clamped_int(raw: string; fallback, lo, hi: int): int {.gcsafe.} =
  let s = raw.strip()
  if s.len == 0:
    return fallback
  try:
    clamp(parse_int(s), lo, hi)
  except CatchableError:
    fallback

proc parse_clamped_float(raw: string; fallback, lo, hi: float): float {.gcsafe.} =
  let s = raw.strip()
  if s.len == 0:
    return fallback
  try:
    clamp(parse_float(s), lo, hi)
  except CatchableError:
    fallback

proc make_unique_username(
  room, desired: string;
  exclude_ws = default(WebSocket);
  has_exclude = false;
): string {.gcsafe.} =
  ## Ensure no two users in the same room share a username (case-insensitive).
  var taken = init_hash_set[string]()
  {.gcsafe.}:
    if room in rooms:
      for ws in rooms[room]:
        if has_exclude and ws == exclude_ws:
          continue
        if ws in clients:
          taken.incl(clients[ws].username.to_lower_ascii())

  if desired.to_lower_ascii() notin taken:
    return desired

  let stem = strip_collision_suffixes(desired)
  var suffix = 2
  while true:
    let suffix_str = "-" & $suffix
    let max_base = max_username_len - suffix_str.len
    let base =
      if stem.len > max_base: stem[0 ..< max_base]
      else: stem
    let candidate = base & suffix_str
    if candidate.to_lower_ascii() notin taken:
      return candidate
    inc suffix

func empty_votes(n: int): seq[Table[WebSocket, int]] =
  result = new_seq[Table[WebSocket, int]](n)
  for i in 0 ..< n:
    result[i] = init_table[WebSocket, int]()

func count_goal_votes(
  goal: GoalPopulation
): tuple[total_votes, voted_candidates, liked_candidates: int, stats: seq[VoteStats]] =
  let n = goal.votes.len
  result.total_votes = 0
  result.voted_candidates = 0
  result.liked_candidates = 0
  result.stats = new_seq[VoteStats](n)
  for i in 0 ..< n:
    var up = 0
    var down = 0
    for _, v in goal.votes[i]:
      if v > 0:
        inc up
      elif v < 0:
        inc down
    let score = up - down
    result.stats[i] = (up: up, down: down, score: score)
    result.total_votes += up + down
    if up + down > 0:
      inc result.voted_candidates
    if score > 0:
      inc result.liked_candidates

func effective_min_votes(goal: GoalPopulation; c: RuntimeConfig): int {.inline.} =
  if c.min_votes_override:
    return clamp(c.min_votes, 1, max_round_min_votes)
  let n = max(1, goal.votes.len)
  if goal.params.min_votes > 0:
    return clamp(goal.params.min_votes, 1, max_round_min_votes)
  if c.min_votes_explicit:
    return clamp(c.min_votes, 1, n)
  clamp((n + 1) div 2, 1, n)

func is_public_image_room(room: string; goal: GoalPopulation): bool {.inline.} =
  room == "lobby" and (goal.task.len == 0 or goal.task == "image")

func effective_max_votes(
  goal: GoalPopulation;
  c: RuntimeConfig;
  is_public_image: bool = false;
): int {.inline.} =
  let base = clamp(goal.params.max_votes_per_user, min_votes_per_user, max_votes_per_user)
  if is_public_image:
    let req = clamp(effective_min_votes(goal, c), min_votes_per_user, max(1, goal.votes.len))
    return max(base, clamp(req, min_votes_per_user, max_votes_per_user))
  base

func count_user_active_votes(goal: GoalPopulation; ws: WebSocket): int =
  for i in 0 ..< goal.votes.len:
    if ws in goal.votes[i]:
      inc result

func allow_vote_burst(client: var ClientInfo; ts_ms: int64): bool =
  var kept: seq[int64] = @[]
  for t in client.vote_timestamps_ms:
    if ts_ms - t <= vote_burst_window_ms:
      kept.add(t)
  if kept.len >= vote_burst_max_ops:
    client.vote_timestamps_ms = kept
    return false
  kept.add(ts_ms)
  client.vote_timestamps_ms = kept
  true

func goal_phase_pipeline(goal: GoalPopulation; tc: TaskConfig): seq[string] =
  ## Declarative ordered phases for a single evolutionary generation.
  result = @[]
  if tc.sim_phase_enabled:
    result.add("simulating")
  result.add("voting")
  if goal.params.brood_size > 1 or goal.phase == "brood":
    result.add("brood")

func initial_phase_for_task(tc: TaskConfig): string {.inline.} =
  if tc.sim_phase_enabled: "simulating" else: "voting"

func clear_goal_brood(goal: var GoalPopulation) {.inline.} =
  goal.brood_size_active = 0
  goal.brood_pop_size = 0
  goal.brood_top_score = 0
  goal.brood_fixed = @[]
  goal.brood_groups = @[]

func count_goal_brood_progress(
  goal: GoalPopulation
): tuple[decided, total_votes, total_broods, required_broods: int] =
  result.total_broods = goal.brood_groups.len
  if result.total_broods == 0:
    return
  result.required_broods = min(result.total_broods, max_required_broods)
  for grp in goal.brood_groups:
    let vc = grp.votes.len
    if vc > 0:
      inc result.decided
      result.total_votes += vc

func pick_brood_winner(grp: BroodGroup): int =
  let k = grp.candidates_json.len
  if k <= 1:
    return 0
  var counts = new_seq[int](k)
  for _, ci in grp.votes:
    if ci >= 0 and ci < k:
      inc counts[ci]
  var best_ci = 0
  var best_cnt = counts[0]
  for ci in 1 ..< k:
    if counts[ci] > best_cnt:
      best_cnt = counts[ci]
      best_ci = ci
  best_ci

func remaining_countdown_ms(goal: GoalPopulation; now: int64): int64 {.inline.} =
  if goal.countdown_ends_at_ms <= 0:
    return 0
  max(0'i64, goal.countdown_ends_at_ms - now)

proc sync_goal_countdown(goal: var GoalPopulation; c: RuntimeConfig; now: int64) {.gcsafe.} =
  if goal.evolving or goal.phase == "simulating":
    goal.countdown_ends_at_ms = 0
    return
  if goal.phase == "brood":
    let bp = count_goal_brood_progress(goal)
    if bp.total_broods > 0 and bp.decided >= bp.required_broods:
      if goal.countdown_ends_at_ms == 0:
        goal.countdown_ends_at_ms = now + c.countdown_ms
    else:
      goal.countdown_ends_at_ms = 0
    return
  let counted = count_goal_votes(goal)
  let req = effective_min_votes(goal, c)
  let ready =
    if goal.params.min_votes > 0:
      counted.total_votes >= req and counted.liked_candidates >= 1
    elif c.min_votes_explicit:
      counted.total_votes >= req
    else:
      counted.voted_candidates >= req and counted.total_votes >= req and counted.liked_candidates >= 1
  if ready:
    if goal.countdown_ends_at_ms == 0:
      goal.countdown_ends_at_ms = now + c.countdown_ms
  else:
    goal.countdown_ends_at_ms = 0

{.push checks: off, stack_trace: off, line_trace: off, optimization: speed.}
proc json_skip_ws(s: string; start_pos: int): int {.inline, gcsafe.} =
  var i = start_pos
  let n = s.len
  while i < n and s[i] in {' ', '\t', '\r', '\n'}:
    inc i
  i

proc c_memchr(s: pointer; c: cint; n: csize_t): pointer {.importc: "memchr", header: "<string.h>", nodecl.}

proc json_skip_string(s: string; open_quote: int): int {.inline, gcsafe.} =
  var pos = open_quote + 1
  let n = s.len
  while pos < n:
    let found = c_memchr(unsafe_addr s[pos], cint('"'), csize_t(n - pos))
    if found == nil: return -1
    let q = pos + (cast[int](found) - cast[int](unsafe_addr s[pos]))
    var bs = 0
    var k = q - 1
    while k >= pos and s[k] == '\\':
      inc bs
      dec k
    if (bs and 1) == 0:
      return q
    pos = q + 1
  -1

proc json_find_value_end(s: string; start_pos: int): int {.gcsafe.} =
  let n = s.len
  var i = json_skip_ws(s, start_pos)
  if i >= n:
    return -1
  let first = s[i]
  if first == '"':
    return json_skip_string(s, i)
  elif first in {'{', '['}:
    var depth = 1
    inc i
    while i < n:
      let ch = s[i]
      if ch == '"':
        i = json_skip_string(s, i)
        if i < 0: return -1
      elif ch in {'{', '['}:
        inc depth
      elif ch in {'}', ']'}:
        dec depth
        if depth == 0:
          return i
      inc i
    return -1
  else:
    while i < n and s[i] notin {',', '}', ']', ' ', '\t', '\r', '\n'}:
      inc i
    return i - 1

iterator walk_json_fields(obj_json: string): tuple[key, span, val: Slice[int]] =
  let n = obj_json.len
  var i = json_skip_ws(obj_json, 0)
  if i < n and obj_json[i] == '{':
    inc i
    while true:
      i = json_skip_ws(obj_json, i)
      if i >= n or obj_json[i] == '}': break
      let k_end = json_find_value_end(obj_json, i)
      if k_end <= i or obj_json[i] != '"': break
      let colon = json_skip_ws(obj_json, k_end + 1)
      if colon >= n or obj_json[colon] != ':': break
      let v_start = json_skip_ws(obj_json, colon + 1)
      let v_end = json_find_value_end(obj_json, v_start)
      if v_end < v_start: break
      yield (key: (i + 1) .. (k_end - 1), span: i .. v_end, val: v_start .. v_end)
      i = json_skip_ws(obj_json, v_end + 1)
      if i < n and obj_json[i] == ',': inc i

proc json_split_array_items(arr_json: string): seq[string] {.gcsafe.} =
  result = @[]
  let n = arr_json.len
  var i = json_skip_ws(arr_json, 0)
  if i >= n or arr_json[i] != '[':
    return
  inc i
  while true:
    i = json_skip_ws(arr_json, i)
    if i >= n or arr_json[i] == ']': break
    let v_end = json_find_value_end(arr_json, i)
    if v_end < i: break
    result.add(arr_json[i .. v_end])
    i = json_skip_ws(arr_json, v_end + 1)
    if i < n and arr_json[i] == ',': inc i

proc json_strip_object_field(obj_json, field_name: string): string {.gcsafe.} =
  var kept: seq[Slice[int]] = @[]
  var stripped = false
  var cap = 2
  for f in walk_json_fields(obj_json):
    if obj_json[f.key] == field_name:
      stripped = true
    else:
      kept.add(f.span)
      cap += f.span.b - f.span.a + 2
  if not stripped:
    return obj_json
  result = new_string_of_cap(cap)
  result.add('{')
  for idx, sl in kept:
    if idx > 0: result.add(',')
    result.add(obj_json[sl])
  result.add('}')

proc json_detach_goal_genomes(obj_json: string): tuple[lean_obj: string, genomes_raw: string] {.gcsafe.} =
  result = (obj_json, "[]")
  var parts: seq[string] = @[]
  var found = false
  for f in walk_json_fields(obj_json):
    if obj_json[f.key] == "genomes":
      result.genomes_raw = obj_json[f.val]
      parts.add("\"genomes\":[]")
      found = true
    else:
      parts.add(obj_json[f.span])
  if found:
    result.lean_obj = "{" & parts.join(",") & "}"

proc json_detach_goals_array(goals_json: string): tuple[lean_arr: string, genomes_list: seq[string]] {.gcsafe.} =
  let raw_goals = json_split_array_items(goals_json)
  result.genomes_list = new_seq_of_cap[string](raw_goals.len)
  var lean_goals = new_seq_of_cap[string](raw_goals.len)
  for g_raw in raw_goals:
    let det = json_detach_goal_genomes(g_raw)
    lean_goals.add(det.lean_obj)
    result.genomes_list.add(det.genomes_raw)
  result.lean_arr = "[" & lean_goals.join(",") & "]"

proc json_detach_state_doc(state_json: string): tuple[lean_doc: string, genomes_list: seq[string]] {.gcsafe.} =
  result = (state_json, @[])
  var parts: seq[string] = @[]
  var found = false
  for f in walk_json_fields(state_json):
    if state_json[f.key] == "goals":
      let sub = json_detach_goals_array(state_json[f.val])
      result.genomes_list = sub.genomes_list
      parts.add("\"goals\":" & sub.lean_arr)
      found = true
    else:
      parts.add(state_json[f.span])
  if found:
    result.lean_doc = "{" & parts.join(",") & "}"
{.pop.}

proc index_goal_genomes(
  genomes_json: string;
  build_sharded: bool;
): tuple[items: seq[string], sharded_json: string, uuid_map: Table[string, int], head_uuid: string, stubs_arr: JsonNode] {.gcsafe.} =
  ## Single-pass indexer building O(1) per-slot raw JSON slices, UUID lookup map, and adapter-free stubs.
  result.items = @[]
  result.uuid_map = init_table[string, int]()
  result.head_uuid = ""
  result.stubs_arr = new_j_array()
  if genomes_json.len == 0:
    result.sharded_json = if build_sharded: "[]" else: ""
    return
  result.items = json_split_array_items(genomes_json)
  if result.items.len == 0:
    result.sharded_json = if build_sharded: "[]" else: ""
    return
  var stub_parts = if build_sharded: new_seq_of_cap[string](result.items.len) else: @[]
  for idx in 0 ..< result.items.len:
    let stub_str = json_strip_object_field(result.items[idx], "adapters")
    if build_sharded:
      stub_parts.add(stub_str)
    try:
      let stub_node = parse_json(stub_str)
      result.stubs_arr.add(stub_node)
      if stub_node.kind == JObject:
        let u = stub_node.get_or_default("uuid").get_str("").strip().to_lower_ascii()
        if u.len == 8:
          result.uuid_map[u] = idx
          if idx == 0:
            result.head_uuid = u
    except CatchableError:
      result.stubs_arr.add(new_j_object())
  result.sharded_json = if build_sharded: "[" & stub_parts.join(",") & "]" else: ""

proc extract_lineage_nodes_from_stubs(stubs_arr: JsonNode; gen: int): JsonNode {.gcsafe.} =
  result = new_j_array()
  if stubs_arr == nil or stubs_arr.kind != JArray:
    return
  for item in stubs_arr:
    if item.kind != JObject:
      continue
    let gid = item.get_or_default("id").get_int(0)
    let guuid = item.get_or_default("uuid").get_str("")
    let raw_parents = item.get_or_default("parents")
    let parents_node =
      if raw_parents != nil and raw_parents.kind == JArray: raw_parents
      else: new_j_array()
    let op_str = item.get_or_default("op").get_str(if gen == 0: "random" else: "mutation")
    if item.get_or_default("kind").get_str("") == "populora":
      result.add(%*{
        "id": gid,
        "uuid": guuid,
        "gen": gen,
        "parents": parents_node,
        "op": op_str,
        "kind": "populora",
        "hash": item.get_or_default("hash").get_str(""),
        "base_hash": item.get_or_default("base_hash").get_str(""),
      })
    else:
      result.add(%*{
        "id": gid,
        "uuid": guuid,
        "gen": gen,
        "parents": parents_node,
        "op": op_str,
        "weights": item.get_or_default("weights"),
        "biases": item.get_or_default("biases"),
        "activations": item.get_or_default("activations"),
      })

proc extract_lineage_nodes(genomes_json: string; gen: int): JsonNode {.gcsafe.} =
  if genomes_json.len == 0:
    return new_j_array()
  let indexed = index_goal_genomes(genomes_json, false)
  extract_lineage_nodes_from_stubs(indexed.stubs_arr, gen)

proc compute_lean_lineage(lineage_json, genomes_json: string; gen: int): tuple[lean_json: string, total: int] {.gcsafe.} =
  ## Precompute active-ancestry JSON (omitting live generation weights) once per generation change.
  let parsed =
    if lineage_json.len > 0: parse_json(lineage_json)
    else: extract_lineage_nodes(genomes_json, gen)
  if parsed.kind != JArray:
    return ("[]", 0)
  result.total = parsed.len

  var by_id = init_table[int, JsonNode]()
  for node in parsed:
    by_id[node.get_or_default("id").get_int(0)] = node

  var keep = init_hash_set[int]()
  var queue: seq[int] = @[]
  for node in parsed:
    if node.get_or_default("gen").get_int(-1) == gen:
      let gid = node.get_or_default("id").get_int(0)
      if gid notin keep:
        keep.incl(gid)
        queue.add(gid)

  var q_idx = 0
  while q_idx < queue.len:
    let cur_id = queue[q_idx]
    inc q_idx
    if cur_id in by_id:
      for p in by_id[cur_id].get_or_default("parents"):
        let pid = p.get_int(0)
        if pid notin keep and pid in by_id:
          keep.incl(pid)
          queue.add(pid)

  var arr = new_j_array()
  for node in parsed:
    let gid = node.get_or_default("id").get_int(0)
    if gid notin keep:
      continue
    if node.get_or_default("gen").get_int(-1) == gen:
      var lean = copy(node)
      if "weights" in lean: lean.delete("weights")
      if "biases" in lean: lean.delete("biases")
      if "activations" in lean: lean.delete("activations")
      if "adapters" in lean: lean.delete("adapters")
      arr.add(lean)
    else:
      arr.add(node)
  result.lean_json = $arr

proc refresh_goal_lineage_cache(goal: var GoalPopulation) {.gcsafe.} =
  let cached = compute_lean_lineage(goal.lineage_json, goal.genomes_json, goal.generation)
  goal.lean_lineage_json = cached.lean_json
  goal.lineage_total = cached.total

proc lineage_payload_node(goal: GoalPopulation; full: bool): tuple[lineage: JsonNode, total: int] {.gcsafe.} =
  ## Sends only the current generation's ancestry by default; the client asks
  ## for the archived `all` view on demand.
  result.lineage = new_j_array()
  if not full and goal.lean_lineage_json.len > 0:
    result.lineage = parse_json(goal.lean_lineage_json)
    result.total = goal.lineage_total
    return
  let parsed =
    if goal.lineage_json.len > 0: parse_json(goal.lineage_json)
    else: extract_lineage_nodes(goal.genomes_json, goal.generation)
  if parsed.kind != JArray:
    return
  result.total = parsed.len
  if full:
    for node in parsed:
      result.lineage.add(node)
    return
  let fallback = compute_lean_lineage(goal.lineage_json, goal.genomes_json, goal.generation)
  result.lineage = parse_json(fallback.lean_json)
  result.total = fallback.total

proc merge_missing_lineage_json(prev_lineage_json, genomes_json: string; gen: int): string {.gcsafe.} =
  ## Append any newly added genomes (e.g. from population resize) without duplicating existing IDs.
  let fresh = extract_lineage_nodes(genomes_json, gen)
  var combined =
    if prev_lineage_json.len > 0: parse_json(prev_lineage_json)
    else: new_j_array()
  if combined.kind != JArray:
    combined = new_j_array()
  var existing_ids = init_hash_set[int]()
  for node in combined:
    existing_ids.incl(node.get_or_default("id").get_int(0))
  for node in fresh:
    let gid = node.get_or_default("id").get_int(0)
    if gid notin existing_ids:
      existing_ids.incl(gid)
      combined.add(node)
  $combined

proc merge_lineage_from_fresh(
  prev_lineage_json: string;
  fresh: JsonNode;
  cap_mult: int = lineage_cap_multiplier;
  min_cap: int = max_lineage_nodes;
  max_cap: int = max_lineage_cap;
): string {.gcsafe.} =
  let eff_max_nodes = clamp(fresh.len * cap_mult, min_cap, max_cap)
  var combined =
    if prev_lineage_json.len > 0: parse_json(prev_lineage_json)
    else: new_j_array()
  if combined.kind != JArray:
    combined = new_j_array()
  for node in fresh:
    combined.add(node)

  if combined.len > eff_max_nodes:
    ## Keep ancestor chain reachable from the latest generation, bounded to eff_max_nodes.
    var by_id = init_table[int, JsonNode]()
    for node in combined:
      by_id[node["id"].get_int()] = node
    var keep_ids = init_hash_set[int]()
    var queue: seq[int] = @[]
    for node in fresh:
      let gid = node["id"].get_int()
      if gid notin keep_ids:
        keep_ids.incl(gid)
        queue.add(gid)

    var q_idx = 0
    while q_idx < queue.len:
      let cur_id = queue[q_idx]
      inc q_idx
      if cur_id in by_id:
        for p in by_id[cur_id].get_or_default("parents"):
          let pid = p.get_int()
          if pid notin keep_ids and pid in by_id:
            keep_ids.incl(pid)
            queue.add(pid)

    var pruned = new_j_array()
    for node in combined:
      if node["id"].get_int() in keep_ids:
        pruned.add(node)
    if pruned.len > eff_max_nodes:
      var trimmed = new_j_array()
      let start_idx = pruned.len - eff_max_nodes
      for i in start_idx ..< pruned.len:
        trimmed.add(pruned[i])
      pruned = trimmed
    combined = pruned
  $combined

proc merge_lineage_json(prev_lineage_json, genomes_json: string; gen: int): string {.gcsafe.} =
  merge_lineage_from_fresh(prev_lineage_json, extract_lineage_nodes(genomes_json, gen))

proc build_votes_json(
  goal: GoalPopulation;
  client_map: Table[WebSocket, ClientInfo];
): tuple[votes_arr: JsonNode, total_votes, voted_candidates, liked_candidates: int] {.gcsafe.} =
  let counted = count_goal_votes(goal)
  result.total_votes = counted.total_votes
  result.voted_candidates = counted.voted_candidates
  result.liked_candidates = counted.liked_candidates
  result.votes_arr = new_j_array()
  for i in 0 ..< goal.votes.len:
    let s = counted.stats[i]
    if goal.votes[i].len == 0:
      result.votes_arr.add(%*{
        "up": 0,
        "down": 0,
        "score": 0,
        "up_users": new_j_array(),
        "down_users": new_j_array(),
      })
      continue
    var up_users: seq[string] = @[]
    var down_users: seq[string] = @[]
    for ws, v in goal.votes[i]:
      if ws in client_map:
        if v > 0:
          up_users.add(client_map[ws].username)
        elif v < 0:
          down_users.add(client_map[ws].username)
    if up_users.len > 1:
      up_users.sort(cmp_ignore_case)
    if down_users.len > 1:
      down_users.sort(cmp_ignore_case)
    result.votes_arr.add(%*{
      "up": s.up,
      "down": s.down,
      "score": s.score,
      "up_users": up_users,
      "down_users": down_users,
    })

proc build_brood_votes_arr(
  goal: GoalPopulation;
  client_map: Table[WebSocket, ClientInfo];
): JsonNode {.gcsafe.} =
  result = new_j_array()
  for grp in goal.brood_groups:
    let k = grp.candidates_json.len
    var counts = new_seq[int](k)
    var voters = new_seq[seq[string]](k)
    for i in 0 ..< k:
      voters[i] = @[]
    for ws, ci in grp.votes:
      if ci >= 0 and ci < k:
        inc counts[ci]
        if ws in client_map:
          voters[ci].add(client_map[ws].username)
    var counts_j = new_j_array()
    var voters_j = new_j_array()
    var total_v = 0
    for i in 0 ..< k:
      total_v += counts[i]
      counts_j.add(%counts[i])
      if voters[i].len > 1:
        voters[i].sort(cmp_ignore_case)
      voters_j.add(%voters[i])
    let winner = if total_v > 0: pick_brood_winner(grp) else: -1
    result.add(%*{
      "counts": counts_j,
      "voters": voters_j,
      "total": total_v,
      "leader": winner,
    })

proc build_brood_payload_str(
  goal: GoalPopulation;
  client_map: Table[WebSocket, ClientInfo];
): string {.gcsafe.} =
  if goal.phase != "brood" or goal.brood_groups.len == 0:
    return "null"
  let votes_arr = build_brood_votes_arr(goal, client_map)
  let bp = count_goal_brood_progress(goal)
  var cap = 128
  for grp in goal.brood_groups:
    for c in grp.candidates_json:
      cap += c.len + 16
  result = new_string_of_cap(cap)
  result.add("{\"brood_size\":")
  result.add($max(2, goal.brood_size_active))
  result.add(",\"fixed_count\":")
  result.add($goal.brood_fixed.len)
  result.add(",\"decided\":")
  result.add($bp.decided)
  result.add(",\"total\":")
  result.add($bp.total_broods)
  result.add(",\"required\":")
  result.add($bp.required_broods)
  result.add(",\"groups\":[")
  for bi, grp in goal.brood_groups:
    if bi > 0: result.add(",")
    result.add("{\"index\":")
    result.add($bi)
    result.add(",\"slot\":")
    result.add($grp.slot)
    result.add(",\"op\":")
    result.add(escape_json(grp.op))
    result.add(",\"parents\":")
    result.add($(%grp.parents))
    result.add(",\"candidates\":[")
    for ci, cjson in grp.candidates_json:
      if ci > 0: result.add(",")
      result.add(cjson)
    result.add("],\"votes\":")
    result.add($votes_arr[bi])
    result.add("}")
  result.add("]}")

proc attach_round_fields(
  obj: JsonNode;
  goal: GoalPopulation;
  c: RuntimeConfig;
  client_map: Table[WebSocket, ClientInfo];
  now: int64;
  room_user_count: int = 1;
  is_public: bool = false;
  include_votes: bool = true;
) {.gcsafe.} =
  let is_pub_image = is_public and (goal.task.len == 0 or goal.task == "image")
  if include_votes:
    let vinfo = build_votes_json(goal, client_map)
    obj["votes"] = vinfo.votes_arr
    obj["total_votes"] = %vinfo.total_votes
    obj["voted_candidates"] = %vinfo.voted_candidates
    obj["liked_candidates"] = %vinfo.liked_candidates
  else:
    let counted = count_goal_votes(goal)
    obj["votes"] = new_j_array()
    obj["total_votes"] = %counted.total_votes
    obj["voted_candidates"] = %counted.voted_candidates
    obj["liked_candidates"] = %counted.liked_candidates
  obj["min_votes"] = %effective_min_votes(goal, c)
  obj["max_votes_per_user"] = %effective_max_votes(goal, c, is_pub_image)
  let tc = task_cfg_locked(goal.task)
  let phases = goal_phase_pipeline(goal, tc)
  let eff_phase =
    if goal.evolving: "evolving"
    elif goal.phase.len > 0: goal.phase
    else: initial_phase_for_task(tc)
  let phase_idx =
    if eff_phase == "evolving": max(0, phases.len - 1)
    else: max(0, phases.find(eff_phase))
  let is_sharded = tc.sim_phase_enabled and goal.votes.len > max(3, tc.crowd_shard_threshold)
  obj["phase"] = %eff_phase
  if include_votes or phases.len > 1:
    obj["phases"] = %phases
    obj["phase_index"] = %phase_idx
    obj["phase_count"] = %phases.len
  obj["sim_phase_enabled"] = %tc.sim_phase_enabled
  obj["sim_completed"] = %goal.sim_completed
  obj["sim_total"] = %max(1, goal.votes.len)
  obj["auto_evolve"] = %goal.auto_evolve
  obj["genomes_sharded"] = %is_sharded
  obj["crowd_shard_threshold"] = %max(3, tc.crowd_shard_threshold)
  obj["countdown_ms"] = %remaining_countdown_ms(goal, now)
  obj["countdown_duration_ms"] = %c.countdown_ms
  obj["evolving"] = %goal.evolving
  obj["mutation_sigma"] = %goal.params.mut_sigma
  obj["elite_frac"] = %goal.params.elite_frac
  obj["crossover_rate"] = %goal.params.crossover_rate
  if goal.params.brood_size > 1 or goal.phase == "brood":
    obj["brood_size"] = %goal.params.brood_size
  if goal.phase == "brood" and goal.brood_groups.len > 0:
    let bp = count_goal_brood_progress(goal)
    obj["brood_decided"] = %bp.decided
    obj["brood_total"] = %bp.total_broods
    obj["brood_required"] = %bp.required_broods
  obj["population_size"] = %goal.votes.len
  obj["max_population_size"] = %tc.max_pop_size

proc goal_to_json_str(
  goal: GoalPopulation;
  c: RuntimeConfig;
  client_map: Table[WebSocket, ClientInfo];
  now: int64;
  room_user_count: int = 1;
  is_public: bool = false;
  include_genomes: bool = true;
  defer_adapters: bool = false;
  defer_lineage: bool = false;
): string {.gcsafe.} =
  ## Fast zero-reparse serializer: splices pre-serialized `genomes_json` (or `sharded_genomes_json`) directly.
  let round_obj = new_j_object()
  round_obj.attach_round_fields(
    goal, c, client_map, now, room_user_count, is_public, include_votes = include_genomes
  )
  let round_str = $round_obj
  let tc = task_cfg_locked(goal.task)
  let is_sharded = tc.sim_phase_enabled and goal.votes.len > max(3, tc.crowd_shard_threshold)
  let strip_adapters = include_genomes and (is_sharded or defer_adapters) and goal.sharded_genomes_json.len > 0
  let genomes_str =
    if not include_genomes: "[]"
    elif strip_adapters: goal.sharded_genomes_json
    elif goal.genomes_json.len > 0: goal.genomes_json
    else: "[]"
  let adapters_loaded = include_genomes and (not strip_adapters)
  let env_params_str = if goal.env_params_json.len > 0: goal.env_params_json else: "{}"
  let include_lineage = include_genomes and (not defer_lineage)
  let lineage_str =
    if include_lineage and goal.lean_lineage_json.len > 0: goal.lean_lineage_json
    else: "[]"
  let brood_str =
    if include_genomes and goal.phase == "brood" and goal.brood_groups.len > 0:
      build_brood_payload_str(goal, client_map)
    else: "null"
  let task_str = if goal.task.len > 0: goal.task else: "image"

  result = new_string_of_cap(genomes_str.len + lineage_str.len + env_params_str.len + brood_str.len + round_str.len + 304)
  result.add("{\"name\":")
  result.add(escape_json(goal.name))
  result.add(",\"task\":")
  result.add(escape_json(task_str))
  result.add(",\"env_params\":")
  result.add(env_params_str)
  result.add(",\"generation\":")
  result.add($goal.generation)
  result.add(",\"genomes_loaded\":")
  result.add(if include_genomes: "true" else: "false")
  result.add(",\"genomes_adapters_loaded\":")
  result.add(if adapters_loaded: "true" else: "false")
  result.add(",\"genomes\":")
  result.add(genomes_str)
  if include_genomes:
    result.add(",\"brood\":")
    result.add(brood_str)
  result.add(",\"lineage_loaded\":")
  result.add(if include_lineage: "true" else: "false")
  result.add(",\"lineage\":")
  result.add(lineage_str)
  result.add(",\"lineage_total\":")
  result.add($goal.lineage_total)
  result.add(",")
  result.add(round_str[1 .. ^1])

proc build_brood_votes_updated_payload(
  goal: GoalPopulation;
  c: RuntimeConfig;
  client_map: Table[WebSocket, ClientInfo];
  now: int64;
  room_user_count: int = 1;
  is_public: bool = false;
): string {.gcsafe.} =
  let node = %*{
    "type": "brood_votes_updated",
    "goal": goal.name,
    "generation": goal.generation,
    "brood_votes": build_brood_votes_arr(goal, client_map),
  }
  node.attach_round_fields(goal, c, client_map, now, room_user_count, is_public, include_votes = false)
  $node

proc build_votes_updated_payload(
  goal: GoalPopulation;
  c: RuntimeConfig;
  client_map: Table[WebSocket, ClientInfo];
  now: int64;
  room_user_count: int = 1;
  is_public: bool = false;
): string {.gcsafe.} =
  let node = %*{
    "type": "votes_updated",
    "goal": goal.name,
  }
  node.attach_round_fields(goal, c, client_map, now, room_user_count, is_public)
  $node

proc build_goal_updated_payload(
  goal: GoalPopulation;
  c: RuntimeConfig;
  client_map: Table[WebSocket, ClientInfo];
  now: int64;
  room_user_count: int = 1;
  is_public: bool = false;
  include_genomes: bool = true;
  defer_adapters: bool = false;
  defer_lineage: bool = false;
): string {.gcsafe.} =
  let g_str = goal_to_json_str(
    goal, c, client_map, now, room_user_count,
    is_public = is_public,
    include_genomes = include_genomes,
    defer_adapters = defer_adapters,
    defer_lineage = defer_lineage,
  )
  result = new_string_of_cap(g_str.len + 32)
  result.add("{\"type\":\"goal_updated\",\"goal\":")
  result.add(g_str)
  result.add("}")

func resolve_lazy_target_goal(evo: RoomEvolution; focus_goal: string): string =
  if focus_goal.len > 0 and focus_goal in evo.goals:
    return focus_goal
  for name in evo.goal_order:
    if name in evo.goals:
      return name
  ""

proc build_evolution_state_payload(
  room: string;
  lazy_goals: bool = false;
  focus_goal: string = "";
  defer_adapters: bool = false;
  defer_lineage: bool = false;
): string {.gcsafe.} =
  let now = now_ms()
  let is_pub = room == "lobby"
  {.gcsafe.}:
    let r_cnt = if room in rooms: max(1, rooms[room].len) else: 1
    var parts: seq[string] = @[]
    var total_len = 36
    if room in room_evolution:
      let evo = room_evolution[room]
      let active_target = if lazy_goals: resolve_lazy_target_goal(evo, focus_goal) else: ""
      for name in evo.goal_order:
        if name in evo.goals:
          let inc_genomes = (not lazy_goals) or (name == active_target)
          let g_str = goal_to_json_str(
            evo.goals[name], cfg, clients, now, r_cnt,
            is_public = is_pub,
            include_genomes = inc_genomes,
            defer_adapters = defer_adapters,
            defer_lineage = defer_lineage,
          )
          total_len += g_str.len + 1
          parts.add(g_str)
    result = new_string_of_cap(total_len)
    result.add("{\"type\":\"evolution_state\",\"goals\":[")
    for idx, p in parts:
      if idx > 0:
        result.add(",")
      result.add(p)
    result.add("]}")

proc goal_params_json(p: GoalParams): JsonNode {.gcsafe.} =
  %*{
    "max_votes_per_user": p.max_votes_per_user,
    "min_votes": p.min_votes,
    "mut_sigma": p.mut_sigma,
    "elite_frac": p.elite_frac,
    "crossover_rate": p.crossover_rate,
    "brood_size": p.brood_size,
  }

proc parse_goal_params_json(node: JsonNode; fallback: GoalParams): GoalParams {.gcsafe.} =
  if node == nil or node.kind != JObject:
    return fallback
  clamp_goal_params(GoalParams(
    max_votes_per_user: node.get_or_default("max_votes_per_user").get_int(fallback.max_votes_per_user),
    min_votes: node.get_or_default("min_votes").get_int(fallback.min_votes),
    mut_sigma: node.get_or_default("mut_sigma").get_float(fallback.mut_sigma),
    elite_frac: node.get_or_default("elite_frac").get_float(fallback.elite_frac),
    crossover_rate: node.get_or_default("crossover_rate").get_float(fallback.crossover_rate),
    brood_size: node.get_or_default("brood_size").get_int(fallback.brood_size),
  ))

proc serialize_brood_state_str(g: GoalPopulation): string {.gcsafe.} =
  if g.phase != "brood" or g.brood_groups.len == 0:
    return ""
  var cap = 128
  for f in g.brood_fixed: cap += f.genome_json.len + 24
  for grp in g.brood_groups:
    for c in grp.candidates_json: cap += c.len + 16
  result = new_string_of_cap(cap)
  result.add("{\"brood_size\":")
  result.add($max(2, g.brood_size_active))
  result.add(",\"pop_size\":")
  result.add($max(1, g.brood_pop_size))
  result.add(",\"top_score\":")
  result.add($g.brood_top_score)
  result.add(",\"fixed\":[")
  for i, f in g.brood_fixed:
    if i > 0: result.add(",")
    result.add("{\"slot\":")
    result.add($f.slot)
    result.add(",\"genome\":")
    result.add(f.genome_json)
    result.add("}")
  result.add("],\"broods\":[")
  for bi, grp in g.brood_groups:
    if bi > 0: result.add(",")
    result.add("{\"slot\":")
    result.add($grp.slot)
    result.add(",\"op\":")
    result.add(escape_json(grp.op))
    result.add(",\"parents\":")
    result.add($(%grp.parents))
    result.add(",\"candidates\":[")
    for ci, cjson in grp.candidates_json:
      if ci > 0: result.add(",")
      result.add(cjson)
    result.add("]}")
  result.add("]}")

proc parse_brood_bundle_node(
  node: JsonNode;
  fallback_pop: int;
  fallback_brood_size: int;
): tuple[
  ok: bool,
  brood_size: int,
  pop_size: int,
  top_score: int,
  fixed: seq[BroodFixedSlot],
  groups: seq[BroodGroup],
] {.gcsafe.} =
  if node == nil or node.kind != JObject:
    return
  result.brood_size = clamp(node.get_or_default("brood_size").get_int(fallback_brood_size), min_brood_size, max_brood_size)
  result.pop_size = max(1, node.get_or_default("pop_size").get_int(fallback_pop))
  result.top_score = node.get_or_default("top_score").get_int(0)
  result.fixed = @[]
  result.groups = @[]
  let fixed_node = node.get_or_default("fixed")
  if fixed_node != nil and fixed_node.kind == JArray:
    for item in fixed_node:
      if item.kind == JObject and item.has_key("genome"):
        result.fixed.add(BroodFixedSlot(
          slot: max(0, item.get_or_default("slot").get_int(result.fixed.len)),
          genome_json: $item["genome"],
        ))
  let broods_node = node.get_or_default("broods")
  if broods_node != nil and broods_node.kind == JArray:
    for item in broods_node:
      if item.kind != JObject:
        continue
      let cands_node = item.get_or_default("candidates")
      if cands_node == nil or cands_node.kind != JArray or cands_node.len == 0:
        continue
      var cands: seq[string] = @[]
      for c in cands_node:
        cands.add($c)
      var pids: seq[int] = @[]
      let pnode = item.get_or_default("parents")
      if pnode != nil and pnode.kind == JArray:
        for p in pnode:
          pids.add(p.get_int(0))
      result.groups.add(BroodGroup(
        slot: max(0, item.get_or_default("slot").get_int(0)),
        parents: pids,
        op: item.get_or_default("op").get_str("crossover"),
        candidates_json: cands,
        votes: init_table[WebSocket, int](),
      ))
  result.ok = result.fixed.len > 0 or result.groups.len > 0

proc serialize_room_evolution(evo: RoomEvolution): string {.gcsafe.} =
  let dp_str = $goal_params_json(evo.default_params)
  var parts: seq[string] = @[]
  var total_len = dp_str.len + 36
  for name in evo.goal_order:
    if name notin evo.goals:
      continue
    let g = evo.goals[name]
    let g_genomes = if g.genomes_json.len > 0: g.genomes_json else: "[]"
    let g_lineage =
      if g.lineage_json.len > 0: g.lineage_json
      else: $extract_lineage_nodes(g.genomes_json, g.generation)
    let gp_str = $goal_params_json(g.params)
    let env_p_str = if g.env_params_json.len > 0: g.env_params_json else: "{}"
    let brood_state_str = serialize_brood_state_str(g)
    let task_str = if g.task.len > 0: g.task else: "image"
    var item = new_string_of_cap(g_genomes.len + g_lineage.len + gp_str.len + env_p_str.len + brood_state_str.len + 168)
    item.add("{\"name\":")
    item.add(escape_json(g.name))
    item.add(",\"task\":")
    item.add(escape_json(task_str))
    item.add(",\"env_params\":")
    item.add(env_p_str)
    item.add(",\"generation\":")
    item.add($g.generation)
    item.add(",\"genomes\":")
    item.add(g_genomes)
    item.add(",\"lineage\":")
    item.add(g_lineage)
    item.add(",\"params\":")
    item.add(gp_str)
    item.add(",\"phase\":")
    item.add(escape_json(if g.phase.len > 0: g.phase else: "voting"))
    item.add(",\"sim_completed\":")
    item.add($g.sim_completed)
    item.add(",\"auto_evolve\":")
    item.add(if g.auto_evolve: "true" else: "false")
    if brood_state_str.len > 0:
      item.add(",\"brood_state\":")
      item.add(brood_state_str)
    item.add("}")
    total_len += item.len + 1
    parts.add(item)
  result = new_string_of_cap(total_len)
  result.add("{\"default_params\":")
  result.add(dp_str)
  result.add(",\"goals\":[")
  for idx, p in parts:
    if idx > 0:
      result.add(",")
    result.add(p)
  result.add("]}")

proc build_presence_payload(room: string): string {.gcsafe.} =
  var users: seq[string] = @[]
  {.gcsafe.}:
    if room in rooms:
      for ws in rooms[room]:
        if ws in clients:
          users.add(clients[ws].username)
  users.sort(cmp_ignore_case)
  $(%*{
    "type": "presence",
    "users": users,
    "count": users.len,
  })

proc build_history_payload(room: string; limit: int = history_limit): string {.gcsafe.} =
  var parts: seq[string] = @[]
  {.gcsafe.}:
    if room in room_history:
      parts = room_history[room]
  let total = parts.len
  let eff_limit = clamp(limit, 0, history_limit)
  let start_idx = if eff_limit < total: total - eff_limit else: 0
  let slice =
    if eff_limit <= 0 or total == 0: @[]
    else: parts[start_idx .. ^1]
  let has_more = if start_idx > 0: "true" else: "false"
  "{\"type\":\"history\",\"total\":" & $total & ",\"offset\":" & $start_idx &
    ",\"has_more\":" & has_more & ",\"messages\":[" & slice.join(",") & "]}"

proc system_msg_payload(msg: string): string {.gcsafe.} =
  $(%*{
    "type": "system",
    "message": msg,
    "timestamp": utc_time_str(),
  })

proc room_targets(
  room: string;
  exclude_ws = default(WebSocket);
  has_exclude = false;
): seq[WebSocket] {.gcsafe.} =
  result = @[]
  {.gcsafe.}:
    if room in rooms:
      for ws in rooms[room]:
        if has_exclude and ws == exclude_ws:
          continue
        result.add(ws)

proc queue_evolution_state_outbox_locked(room: string; outbox: var seq[PendingBroadcast]) {.gcsafe.} =
  {.gcsafe.}:
    let targets = room_targets(room)
    if targets.len == 0:
      return
    type EvoBucketKey = tuple[lazy: bool, focus: string, defer_adap: bool, defer_lin: bool]
    var buckets = init_table[EvoBucketKey, seq[WebSocket]]()
    let has_evo = room in room_evolution
    for ws in targets:
      if ws in clients:
        let cinfo = clients[ws]
        var resolved = ""
        if cinfo.lazy_goals:
          resolved =
            if has_evo: resolve_lazy_target_goal(room_evolution[room], cinfo.active_goal)
            else: ""
          clients[ws].active_goal = resolved
        let k: EvoBucketKey = (cinfo.lazy_goals, resolved, cinfo.defer_adapters, cinfo.defer_lineage)
        if k notin buckets:
          buckets[k] = @[]
        buckets[k].add(ws)
      else:
        let k: EvoBucketKey = (false, "", false, false)
        if k notin buckets:
          buckets[k] = @[]
        buckets[k].add(ws)
    for k, bucket_targets in buckets:
      outbox.add(PendingBroadcast(
        targets: bucket_targets,
        payload: build_evolution_state_payload(
          room,
          lazy_goals = k.lazy,
          focus_goal = k.focus,
          defer_adapters = k.defer_adap,
          defer_lineage = k.defer_lin,
        ),
      ))

proc queue_goal_updated_outbox_locked(
  room: string;
  goal: GoalPopulation;
  now: int64;
  outbox: var seq[PendingBroadcast];
) {.gcsafe.} =
  {.gcsafe.}:
    let targets = room_targets(room)
    if targets.len == 0:
      return
    let r_cnt = max(1, targets.len)
    let is_pub = room == "lobby"
    type GoalBucketKey = tuple[inc_genomes: bool, defer_adap: bool, defer_lin: bool]
    var buckets = init_table[GoalBucketKey, seq[WebSocket]]()
    for ws in targets:
      if ws in clients:
        let cinfo = clients[ws]
        let is_stub = cinfo.lazy_goals and cinfo.active_goal != goal.name
        let k: GoalBucketKey = (
          not is_stub,
          (not is_stub) and cinfo.defer_adapters,
          (not is_stub) and cinfo.defer_lineage,
        )
        if k notin buckets:
          buckets[k] = @[]
        buckets[k].add(ws)
      else:
        let k: GoalBucketKey = (true, false, false)
        if k notin buckets:
          buckets[k] = @[]
        buckets[k].add(ws)
    for k, bucket_targets in buckets:
      outbox.add(PendingBroadcast(
        targets: bucket_targets,
        payload: build_goal_updated_payload(
          goal, cfg, clients, now, r_cnt,
          is_public = is_pub,
          include_genomes = k.inc_genomes,
          defer_adapters = k.defer_adap,
          defer_lineage = k.defer_lin,
        ),
      ))

proc client_uses_gzip(ws: WebSocket): bool {.gcsafe.} =
  with_state_lock:
    result = ws in clients and clients[ws].gzip

proc compress_payload(payload: string): string {.gcsafe.} =
  try:
    result = compress(payload, DefaultCompression, df_gzip)
  except CatchableError:
    result = ""

proc send_payload(ws: WebSocket; payload: string) {.gcsafe.} =
  if payload.len >= gzip_min_bytes and client_uses_gzip(ws):
    let gz = compress_payload(payload)
    if gz.len > 0:
      ws.send(gz, BinaryMessage)
      return
  ws.send(payload)

proc broadcast(targets: seq[WebSocket]; payload: string) {.gcsafe.} =
  if payload.len < gzip_min_bytes:
    for ws in targets:
      ws.send(payload)
    return

  var plain: seq[WebSocket] = @[]
  var gz_targets: seq[WebSocket] = @[]
  with_state_lock:
    for ws in targets:
      if ws in clients and clients[ws].gzip:
        gz_targets.add(ws)
      else:
        plain.add(ws)

  for ws in plain:
    ws.send(payload)
  if gz_targets.len > 0:
    let gz = compress_payload(payload)
    if gz.len > 0:
      for ws in gz_targets:
        ws.send(gz, BinaryMessage)
    else:
      for ws in gz_targets:
        ws.send(payload)

proc init_goal_population_locked(
  name, task, genomes_json: string;
  pop_len: int;
  gp: GoalParams;
  auto_evolve_override: int = -1;
  env_params_json: string = "{}";
): GoalPopulation {.gcsafe.} =
  let clean_task = sanitize_task_name_locked(task)
  let tc = task_cfg_locked(clean_task)
  var params = gp
  clamp_control_min_votes_locked(clean_task, params)
  let indexed = index_goal_genomes(genomes_json, tc.sim_phase_enabled)
  let eff_pop_len = if pop_len > 0: pop_len else: max(1, indexed.items.len)
  let l_json = merge_lineage_from_fresh("", extract_lineage_nodes_from_stubs(indexed.stubs_arr, 0))
  let lean = compute_lean_lineage(l_json, genomes_json, 0)
  let init_phase = if tc.sim_phase_enabled: "simulating" else: "voting"
  let init_auto =
    if auto_evolve_override >= 0: auto_evolve_override > 0
    else: tc.default_auto_evolve
  let eff_env_json = if env_params_json.len > 0: env_params_json else: "{}"
  GoalPopulation(
    name: name,
    task: clean_task,
    generation: 0,
    genomes_json: genomes_json,
    sharded_genomes_json: indexed.sharded_json,
    genome_items: indexed.items,
    uuid_to_idx: indexed.uuid_map,
    head_uuid: indexed.head_uuid,
    env_params_json: eff_env_json,
    lineage_json: l_json,
    lean_lineage_json: lean.lean_json,
    lineage_total: lean.total,
    votes: empty_votes(max(1, eff_pop_len)),
    phase: init_phase,
    sim_completed: 0,
    auto_evolve: init_auto,
    countdown_ends_at_ms: 0,
    evolving: false,
    evolving_started_at_ms: 0,
    params: params,
  )

proc populate_room_goals_locked(
  parsed: JsonNode;
  dp: GoalParams;
  detached_genomes: seq[string] = @[];
): RoomEvolution {.gcsafe.} =
  result = RoomEvolution(
    goal_order: @[],
    goals: init_table[string, GoalPopulation](),
    pending_creates: init_hash_set[string](),
    pending_create_tasks: init_table[string, string](),
    pending_create_params: init_table[string, GoalParams](),
    pending_create_env_params: init_table[string, string](),
    default_params: dp,
    initializing: false,
  )
  for idx in 0 ..< parsed.len:
    let item = parsed[idx]
    let name = item["name"].get_str()
    let task = sanitize_task_name_locked(item.get_or_default("task").get_str("image"))
    let g_json =
      if idx < detached_genomes.len: detached_genomes[idx]
      else: $item["genomes"]
    var gp = dp
    let item_mv = item.get_or_default("min_votes").get_int(-1)
    if item_mv > 0:
      gp.min_votes = clamp(item_mv, 1, max_round_min_votes)
    elif gp.min_votes <= 0:
      gp.min_votes = default_min_votes_for_task_locked(task)
    if item.has_key("elite_frac") and item["elite_frac"].kind in {JFloat, JInt}:
      gp.elite_frac = clamp(item["elite_frac"].get_float(gp.elite_frac), min_elite_frac, max_elite_frac)
    elif task != "image":
      gp.elite_frac = default_elite_frac_for_task_locked(task, dp.elite_frac)
    if item.has_key("brood_size") and item["brood_size"].kind == JInt:
      gp.brood_size = clamp(item["brood_size"].get_int(gp.brood_size), min_brood_size, max_brood_size)
    else:
      gp.brood_size = default_brood_size_for_task_locked(task, dp.brood_size)
    let auto_override =
      if item.has_key("auto_evolve") and item["auto_evolve"].kind == JBool:
        (if item["auto_evolve"].get_bool(false): 1 else: 0)
      else: -1
    let env_node = item.get_or_default("env_params")
    let env_json =
      if env_node != nil and env_node.kind == JObject: $env_node
      else: "{}"
    result.goal_order.add(name)
    result.goals[name] = init_goal_population_locked(name, task, g_json, 0, gp, auto_override, env_json)

proc upgrade_handler(request: Request) {.gcsafe.} =
  let room = request.path_params["room"]
  if not is_valid_room(room):
    request.respond(400, body = "Invalid room name")
    return

  let username = sanitize_username(request.query_params["username"])
  let cid = sanitize_cid(request.query_params["cid"])
  let is_reconnect = request.query_params["reconnect"] == "1"
  let initial_goal = sanitize_goal_name(request.query_params["goal"])
  let raw_focus_goal = sanitize_goal_name(request.query_params["focus"])
  let focus_goal = if raw_focus_goal.len > 0: raw_focus_goal else: initial_goal
  let lazy_goals = request.query_params["lazy"] == "1"
  let defer_adapters = request.query_params["defer_adapters"] == "1"
  let defer_lineage = request.query_params["defer_lineage"] == "1"
  let chat_limit = parse_clamped_int(request.query_params["chat_limit"], history_limit, 0, history_limit)
  let raw_initial_task = request.query_params["task"]
  let wants_gzip = request.query_params["gzip"] == "1"
  let raw_evolve_token = request.query_params["evolve_token"].strip()
  var
    base_cfg: RuntimeConfig
    server_evolve_token: string
    initial_task = "image"
    task_max_pop = max_population_size
    task_default_pop = default_population_size
    task_default_min_votes = 0
    task_default_elite = default_elite_frac
    task_default_brood = default_brood_size
  with_state_lock:
    base_cfg = cfg
    server_evolve_token = evolve_token
    initial_task = sanitize_task_name_locked(raw_initial_task)
    task_max_pop = max_pop_for_task_locked(initial_task)
    task_default_pop = default_pop_for_task_locked(initial_task, base_cfg.population_size)
    task_default_min_votes = default_min_votes_for_task_locked(initial_task)
    task_default_elite = default_elite_frac_for_task_locked(initial_task, base_cfg.params.elite_frac)
    task_default_brood = default_brood_size_for_task_locked(initial_task, base_cfg.params.brood_size)
  let can_evolve =
    server_evolve_token.len > 0 and
    constant_time_eq(raw_evolve_token, server_evolve_token)

  let req_pop = parse_clamped_int(
    request.query_params["pop"],
    task_default_pop, min_population_size, task_max_pop,
  )
  let raw_min_votes = request.query_params["min_votes"]
  let req_min_votes =
    if raw_min_votes.strip().len > 0:
      parse_clamped_int(raw_min_votes, task_default_min_votes, 1, max_round_min_votes)
    else:
      task_default_min_votes
  let req_params = GoalParams(
    max_votes_per_user: parse_clamped_int(
      request.query_params["max_votes"],
      base_cfg.params.max_votes_per_user, min_votes_per_user, max_votes_per_user,
    ),
    min_votes: req_min_votes,
    mut_sigma: parse_clamped_float(
      request.query_params["sigma"],
      base_cfg.params.mut_sigma, min_mut_sigma, max_mut_sigma,
    ),
    elite_frac: parse_clamped_float(
      request.query_params["elite_frac"],
      task_default_elite, min_elite_frac, max_elite_frac,
    ),
    crossover_rate: parse_clamped_float(
      request.query_params["crossover"],
      base_cfg.params.crossover_rate, min_crossover_rate, max_crossover_rate,
    ),
    brood_size: parse_clamped_int(
      request.query_params["brood_size"],
      task_default_brood, min_brood_size, max_brood_size,
    ),
  )
  let raw_env_q = request.query_params["env_params"].strip()
  var req_env_params_json = "{}"
  if raw_env_q.len > 0 and raw_env_q.len <= 1024:
    try:
      let parsed_env = parse_json(raw_env_q)
      if parsed_env.kind == JObject:
        req_env_params_json = $parsed_env
    except CatchableError:
      discard

  with_state_lock:
    let ws = request.upgrade_to_web_socket()
    inc next_upgrade_seq
    let seq_num = next_upgrade_seq
    pending_upgrades[ws] = PendingClient(
      room: room,
      username: username,
      cid: cid,
      upgrade_seq: seq_num,
      is_reconnect: is_reconnect,
      initial_goal: initial_goal,
      initial_task: initial_task,
      initial_pop_size: req_pop,
      initial_params: req_params,
      initial_env_params_json: req_env_params_json,
      gzip: wants_gzip,
      can_evolve_now: can_evolve,
      lazy_goals: lazy_goals,
      defer_adapters: defer_adapters,
      defer_lineage: defer_lineage,
      chat_limit: chat_limit,
      active_goal: focus_goal,
    )

proc evict_client_locked(
  ws: WebSocket;
  now: int64;
  outbox: var seq[PendingBroadcast];
  silent_leave: bool = false;
): bool {.gcsafe.} =
  {.gcsafe.}:
    pending_upgrades.del(ws)
    if ws notin clients:
      return false
    let info = clients[ws]
    clients.del(ws)
    if info.room in rooms:
      rooms[info.room].excl(ws)
      if rooms[info.room].len == 0:
        rooms.del(info.room)
    var vote_payloads: seq[string] = @[]
    if info.room in room_evolution:
      let r_cnt = if info.room in rooms: max(1, rooms[info.room].len) else: 1
      for g_name in room_evolution[info.room].goal_order:
        var goal = room_evolution[info.room].goals[g_name]
        var changed = false
        var brood_changed = false
        for i in 0 ..< goal.votes.len:
          if ws in goal.votes[i]:
            goal.votes[i].del(ws)
            changed = true
        for bi in 0 ..< goal.brood_groups.len:
          if ws in goal.brood_groups[bi].votes:
            goal.brood_groups[bi].votes.del(ws)
            brood_changed = true
        if changed or brood_changed:
          goal.sync_goal_countdown(cfg, now)
          room_evolution[info.room].goals[g_name] = goal
          if changed:
            vote_payloads.add(
              build_votes_updated_payload(goal, cfg, clients, now, r_cnt, is_public = (info.room == "lobby"))
            )
          if brood_changed and goal.phase == "brood":
            vote_payloads.add(
              build_brood_votes_updated_payload(goal, cfg, clients, now, r_cnt, is_public = (info.room == "lobby"))
            )
    let targets = room_targets(info.room)
    if targets.len > 0:
      outbox.add(PendingBroadcast(
        targets: targets,
        payload: $(%*{"type": "cursor_leave", "username": info.username}),
      ))
      if not silent_leave:
        outbox.add(PendingBroadcast(
          targets: targets,
          payload: system_msg_payload(info.username & " left"),
        ))
      outbox.add(PendingBroadcast(
        targets: targets,
        payload: build_presence_payload(info.room),
      ))
      for vp in vote_payloads:
        outbox.add(PendingBroadcast(targets: targets, payload: vp))
    true

proc on_open(ws: WebSocket) {.gcsafe.} =
  var
    identity_payload: string
    history_payload: string
    evolution_payload = ""
    presence_payload: string
    join_payload = ""
    targets: seq[WebSocket]
    stale_sockets: seq[WebSocket] = @[]
    extra_outbox: seq[PendingBroadcast] = @[]
    found = false
    superseded = false
    has_evolution_ready = false
    prioritize_evolution = false

  with_state_lock:
    block open_locked:
      if ws notin pending_upgrades:
        break open_locked
      let info = pending_upgrades[ws]
      pending_upgrades.del(ws)
      let now = now_ms()
      var replaced_same_room = false
      var prev_same_room_name = ""
      var same_room_old_sockets: seq[WebSocket] = @[]

      if info.cid.len >= 8:
        var matching_old: seq[WebSocket] = @[]
        for existing_ws, existing_info in clients:
          if existing_ws == ws or existing_info.cid != info.cid:
            continue
          if existing_info.upgrade_seq > info.upgrade_seq:
            superseded = true
            break open_locked
          matching_old.add(existing_ws)
        for old_ws in matching_old:
          let old_info = clients[old_ws]
          stale_sockets.add(old_ws)
          if old_info.room == info.room:
            replaced_same_room = true
            prev_same_room_name = old_info.username
            same_room_old_sockets.add(old_ws)
            clients.del(old_ws)
            if info.room in rooms:
              rooms[info.room].excl(old_ws)
          else:
            discard evict_client_locked(old_ws, now, extra_outbox)

      let unique_name = make_unique_username(info.room, info.username)
      var resolved_active_goal = info.active_goal
      if info.lazy_goals and info.room in room_evolution and not room_evolution[info.room].initializing:
        resolved_active_goal = resolve_lazy_target_goal(room_evolution[info.room], info.active_goal)
      clients[ws] = ClientInfo(
        ws: ws,
        room: info.room,
        username: unique_name,
        cid: info.cid,
        upgrade_seq: info.upgrade_seq,
        last_seen_ms: now,
        last_ping_ms: now,
        gzip: info.gzip,
        can_evolve_now: info.can_evolve_now,
        lazy_goals: info.lazy_goals,
        defer_adapters: info.defer_adapters,
        defer_lineage: info.defer_lineage,
        chat_limit: info.chat_limit,
        active_goal: resolved_active_goal,
      )
      if info.room notin rooms:
        rooms[info.room] = init_hash_set[WebSocket]()
      rooms[info.room].incl(ws)

      if same_room_old_sockets.len > 0 and info.room in room_evolution:
        let r_cnt = max(1, rooms[info.room].len)
        for g_name in room_evolution[info.room].goal_order:
          var goal = room_evolution[info.room].goals[g_name]
          var changed = false
          var transferred = false
          var brood_changed = false
          var brood_transferred = false
          for old_ws in same_room_old_sockets:
            for i in 0 ..< goal.votes.len:
              if old_ws in goal.votes[i]:
                let old_v = goal.votes[i][old_ws]
                goal.votes[i].del(old_ws)
                if info.is_reconnect:
                  goal.votes[i][ws] = old_v
                  transferred = true
                else:
                  changed = true
            for bi in 0 ..< goal.brood_groups.len:
              if old_ws in goal.brood_groups[bi].votes:
                let old_ci = goal.brood_groups[bi].votes[old_ws]
                goal.brood_groups[bi].votes.del(old_ws)
                if info.is_reconnect:
                  goal.brood_groups[bi].votes[ws] = old_ci
                  brood_transferred = true
                else:
                  brood_changed = true
          if changed or brood_changed:
            goal.sync_goal_countdown(cfg, now)
            room_evolution[info.room].goals[g_name] = goal
            if changed:
              extra_outbox.add(PendingBroadcast(
                targets: room_targets(info.room, ws, has_exclude = true),
                payload: build_votes_updated_payload(goal, cfg, clients, now, r_cnt, is_public = (info.room == "lobby")),
              ))
            if brood_changed and goal.phase == "brood":
              extra_outbox.add(PendingBroadcast(
                targets: room_targets(info.room, ws, has_exclude = true),
                payload: build_brood_votes_updated_payload(goal, cfg, clients, now, r_cnt, is_public = (info.room == "lobby")),
              ))
          elif transferred or brood_transferred:
            room_evolution[info.room].goals[g_name] = goal
            if prev_same_room_name != unique_name:
              if transferred:
                extra_outbox.add(PendingBroadcast(
                  targets: room_targets(info.room, ws, has_exclude = true),
                  payload: build_votes_updated_payload(goal, cfg, clients, now, r_cnt, is_public = (info.room == "lobby")),
                ))
              if brood_transferred and goal.phase == "brood":
                extra_outbox.add(PendingBroadcast(
                  targets: room_targets(info.room, ws, has_exclude = true),
                  payload: build_brood_votes_updated_payload(goal, cfg, clients, now, r_cnt, is_public = (info.room == "lobby")),
                ))
      if replaced_same_room and prev_same_room_name.len > 0 and prev_same_room_name != unique_name:
        let others = room_targets(info.room, ws, has_exclude = true)
        if others.len > 0:
          extra_outbox.add(PendingBroadcast(
            targets: others,
            payload: $(%*{"type": "cursor_leave", "username": prev_same_room_name}),
          ))

      identity_payload = $(%*{
        "type": "identity",
        "username": unique_name,
        "can_evolve_now": info.can_evolve_now,
      })
      history_payload = build_history_payload(info.room, info.chat_limit)
      prioritize_evolution = info.defer_adapters

      if info.room in room_evolution and not room_evolution[info.room].initializing:
        evolution_payload = build_evolution_state_payload(
          info.room,
          lazy_goals = info.lazy_goals,
          focus_goal = resolved_active_goal,
          defer_adapters = info.defer_adapters,
          defer_lineage = info.defer_lineage,
        )
        has_evolution_ready = true
      elif info.room notin room_evolution:
        let is_private = info.room != "lobby"
        let room_params =
          if is_private: info.initial_params
          else: cfg.params
        let room_pop =
          if is_private: info.initial_pop_size
          else: cfg.population_size
        room_evolution[info.room] = RoomEvolution(
          goal_order: @[],
          goals: init_table[string, GoalPopulation](),
          pending_creates: init_hash_set[string](),
          pending_create_tasks: init_table[string, string](),
          pending_create_params: init_table[string, GoalParams](),
          pending_create_env_params: init_table[string, string](),
          default_params: room_params,
          initializing: true,
        )
        pending_evo_jobs.add(EvoJob(
          kind: EvoJobKind.init_room,
          room: info.room,
          goal: (if is_private: info.initial_goal else: ""),
          task: (if is_private: info.initial_task else: "image"),
          username: unique_name,
          env_params_json: (if is_private: info.initial_env_params_json else: "{}"),
          population_size: room_pop,
          params: room_params,
        ))

      presence_payload = build_presence_payload(info.room)
      if not (replaced_same_room and prev_same_room_name == unique_name):
        join_payload = system_msg_payload(unique_name & " joined")
      targets = room_targets(info.room)
      found = true

  if superseded:
    send_payload(ws, """{"type":"replaced"}""")
    ws.close()
    return

  for sws in stale_sockets:
    send_payload(sws, """{"type":"replaced"}""")
    sws.close()
  for item in extra_outbox:
    broadcast(item.targets, item.payload)
  if found:
    send_payload(ws, identity_payload)
    if prioritize_evolution:
      if has_evolution_ready:
        send_payload(ws, evolution_payload)
      send_payload(ws, history_payload)
    else:
      send_payload(ws, history_payload)
      if has_evolution_ready:
        send_payload(ws, evolution_payload)
    broadcast(targets, presence_payload)
    if join_payload.len > 0:
      broadcast(targets, join_payload)

proc handle_rename(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  let desired_name = sanitize_username(payload.get_or_default("username").get_str(""))
  var
    old_name = ""
    assigned_name = ""
    can_evolve = false
    presence_payload: string
    rename_payload: string
    cursor_leave_payload: string
    vote_payloads: seq[string] = @[]
    other_targets: seq[WebSocket]
    targets: seq[WebSocket]
    found = false
    changed = false

  with_state_lock:
    if ws in clients:
      found = true
      old_name = clients[ws].username
      can_evolve = clients[ws].can_evolve_now
      let room = clients[ws].room
      assigned_name = make_unique_username(room, desired_name, ws, has_exclude = true)
      if assigned_name != old_name:
        clients[ws].username = assigned_name
        let now = now_ms()
        presence_payload = build_presence_payload(room)
        rename_payload = system_msg_payload(old_name & " is now " & assigned_name)
        cursor_leave_payload = $(%*{"type": "cursor_leave", "username": old_name})
        if room in room_evolution:
          let r_cnt = if room in rooms: max(1, rooms[room].len) else: 1
          for g_name in room_evolution[room].goal_order:
            let goal = room_evolution[room].goals[g_name]
            var has_user_vote = false
            for i in 0 ..< goal.votes.len:
              if ws in goal.votes[i]:
                has_user_vote = true
                break
            if has_user_vote:
              vote_payloads.add(
                build_votes_updated_payload(goal, cfg, clients, now, r_cnt, is_public = (room == "lobby"))
              )
            var has_brood_vote = false
            for bi in 0 ..< goal.brood_groups.len:
              if ws in goal.brood_groups[bi].votes:
                has_brood_vote = true
                break
            if has_brood_vote and goal.phase == "brood":
              vote_payloads.add(
                build_brood_votes_updated_payload(goal, cfg, clients, now, r_cnt, is_public = (room == "lobby"))
              )
        targets = room_targets(room)
        other_targets = room_targets(room, ws, has_exclude = true)
        changed = true

  if found:
    send_payload(ws, $(%*{
      "type": "identity",
      "username": assigned_name,
      "can_evolve_now": can_evolve,
    }))
  if changed:
    broadcast(other_targets, cursor_leave_payload)
    broadcast(targets, presence_payload)
    broadcast(targets, rename_payload)
    for vp in vote_payloads:
      broadcast(targets, vp)

proc handle_cursor(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  let x = clamp(payload.get_or_default("x").get_float(0.0), -0.05, 1.05)
  let y = clamp(payload.get_or_default("y").get_float(0.0), -0.05, 1.05)
  let visible = payload.get_or_default("visible").get_bool(true)
  let goal_name = sanitize_goal_name(payload.get_or_default("goal").get_str(""))
  var
    cursor_payload = ""
    other_targets: seq[WebSocket]
    valid = false

  with_state_lock:
    if ws in clients:
      let info = clients[ws]
      cursor_payload = $(%*{
        "type": "cursor",
        "username": info.username,
        "x": x,
        "y": y,
        "visible": visible,
        "goal": goal_name,
      })
      other_targets = room_targets(info.room, ws, has_exclude = true)
      valid = true

  if valid and other_targets.len > 0:
    broadcast(other_targets, cursor_payload)

proc handle_chat(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  var text = payload.get_or_default("message").get_str("").strip()
  if text.len == 0:
    return
  if text.len > max_message_len:
    text = text[0 ..< max_message_len]

  var
    chat_payload: string
    targets: seq[WebSocket]
    valid = false

  with_state_lock:
    if ws in clients:
      let info = clients[ws]
      let ts = utc_time_str()
      let history_item = $(%*{
        "username": info.username,
        "message": text,
        "timestamp": ts,
      })
      if info.room notin room_history:
        room_history[info.room] = @[]
      room_history[info.room].add(history_item)
      if room_history[info.room].len > history_limit:
        let excess = room_history[info.room].len - history_limit
        room_history[info.room] = room_history[info.room][excess .. ^1]

      pending_saves.add(PendingPersist(
        room: info.room,
        username: info.username,
        message: text,
      ))
      chat_payload = $(%*{
        "type": "chat",
        "username": info.username,
        "message": text,
        "timestamp": ts,
      })
      targets = room_targets(info.room)
      valid = true

  if valid:
    broadcast(targets, chat_payload)

proc handle_create_goal(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  let goal_name = sanitize_goal_name(payload.get_or_default("goal").get_str(""))
  if goal_name.len == 0:
    return
  let raw_task = payload.get_or_default("task").get_str("image")

  with_state_lock:
    let task_name = sanitize_task_name_locked(raw_task)
    if ws in clients:
      let info = clients[ws]
      if info.room != "lobby" and
        info.room in room_evolution and
        goal_name notin room_evolution[info.room].goals and
        goal_name notin room_evolution[info.room].pending_creates:
        let d = room_evolution[info.room].default_params
        let default_pop = default_pop_for_task_locked(task_name, cfg.population_size)
        let req_pop = clamp(
          payload.get_or_default("population_size").get_int(default_pop),
          min_population_size, max_pop_for_task_locked(task_name),
        )
        let req_min_votes = payload.get_or_default("min_votes").get_int(default_min_votes_for_task_locked(task_name))
        let task_default_elite = default_elite_frac_for_task_locked(task_name, d.elite_frac)
        let task_default_brood = default_brood_size_for_task_locked(task_name, d.brood_size)
        let g_params = clamp_goal_params(GoalParams(
          max_votes_per_user: payload.get_or_default("max_votes_per_user").get_int(d.max_votes_per_user),
          min_votes: req_min_votes,
          mut_sigma: payload.get_or_default("mutation_sigma").get_float(d.mut_sigma),
          elite_frac: payload.get_or_default("elite_frac").get_float(task_default_elite),
          crossover_rate: payload.get_or_default("crossover_rate").get_float(d.crossover_rate),
          brood_size: payload.get_or_default("brood_size").get_int(task_default_brood),
        ))
        let env_node = payload.get_or_default("env_params")
        let alt_env_node = payload.get_or_default("env_params")
        var req_env_json = "{}"
        if env_node != nil and env_node.kind == JObject:
          req_env_json = $env_node
        elif alt_env_node != nil and alt_env_node.kind == JObject:
          req_env_json = $alt_env_node
        room_evolution[info.room].pending_creates.incl(goal_name)
        room_evolution[info.room].pending_create_tasks[goal_name] = task_name
        room_evolution[info.room].pending_create_params[goal_name] = g_params
        room_evolution[info.room].pending_create_env_params[goal_name] = req_env_json
        clients[ws].active_goal = goal_name
        pending_evo_jobs.add(EvoJob(
          kind: EvoJobKind.create,
          room: info.room,
          goal: goal_name,
          task: task_name,
          username: info.username,
          env_params_json: req_env_json,
          population_size: req_pop,
          params: g_params,
        ))

proc handle_cast_vote(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  let goal_name = payload.get_or_default("goal").get_str("")
  let idx = payload.get_or_default("index").get_int(-1)
  let req_vote = payload.get_or_default("vote").get_int(0)
  let has_req_gen = payload.has_key("generation")
  let req_gen = if has_req_gen: payload.get_or_default("generation").get_int(-1) else: -1
  if idx < 0 or req_vote notin [-1, 0, 1]:
    return

  var
    votes_payload: string
    throttle_payload = ""
    targets: seq[WebSocket]
    updated = false

  with_state_lock:
    if ws in clients:
      var client_info = clients[ws]
      if client_info.room in room_evolution and goal_name in room_evolution[client_info.room].goals:
        var goal = room_evolution[client_info.room].goals[goal_name]
        if goal.phase == "simulating":
          throttle_payload = $(%*{
            "type": "vote_throttled",
            "goal": goal_name,
            "index": idx,
            "max_votes_per_user": effective_max_votes(goal, cfg, is_public_image_room(client_info.room, goal)),
            "message": "Simulations still running — voting unlocks once all candidates are evaluated",
          })
        elif goal.phase == "brood":
          throttle_payload = $(%*{
            "type": "vote_throttled",
            "goal": goal_name,
            "index": idx,
            "max_votes_per_user": effective_max_votes(goal, cfg, is_public_image_room(client_info.room, goal)),
            "message": "Brood selection is active — pick the strongest sibling in each brood",
          })
        elif not goal.evolving and idx < goal.votes.len and (not has_req_gen or req_gen == goal.generation):
          let now = now_ms()
          let prev_vote = goal.votes[idx].get_or_default(ws, 0)
          let next_vote = if prev_vote == req_vote: 0 else: req_vote
          let cap = effective_max_votes(
            goal, cfg, is_public_image_room(client_info.room, goal)
          )
          if prev_vote == 0 and next_vote != 0 and count_user_active_votes(goal, ws) >= cap:
            throttle_payload = $(%*{
              "type": "vote_throttled",
              "goal": goal_name,
              "index": idx,
              "max_votes_per_user": cap,
              "message": "Vote limit reached (" & $cap & " active votes per round — unvote one to switch)",
            })
          elif not allow_vote_burst(client_info, now):
            clients[ws] = client_info
            throttle_payload = $(%*{
              "type": "vote_throttled",
              "goal": goal_name,
              "index": idx,
              "max_votes_per_user": cap,
              "message": "Vote rate limit reached — pause a moment before clicking again",
            })
          else:
            clients[ws] = client_info
            if next_vote == 0:
              goal.votes[idx].del(ws)
            else:
              goal.votes[idx][ws] = next_vote
            goal.sync_goal_countdown(cfg, now)
            room_evolution[client_info.room].goals[goal_name] = goal
            let r_cnt = if client_info.room in rooms: max(1, rooms[client_info.room].len) else: 1
            votes_payload = build_votes_updated_payload(
              goal, cfg, clients, now, r_cnt, is_public = (client_info.room == "lobby")
            )
            targets = room_targets(client_info.room)
            updated = true

  if throttle_payload.len > 0:
    send_payload(ws, throttle_payload)
  if updated:
    broadcast(targets, votes_payload)

proc handle_cast_brood_vote(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  let goal_name = payload.get_or_default("goal").get_str("")
  let brood_idx = payload.get_or_default("brood_index").get_int(-1)
  let cand_idx = payload.get_or_default("candidate_index").get_int(-1)
  let has_req_gen = payload.has_key("generation")
  let req_gen = if has_req_gen: payload.get_or_default("generation").get_int(-1) else: -1
  if brood_idx < 0 or cand_idx < 0:
    return

  var
    votes_payload = ""
    throttle_payload = ""
    targets: seq[WebSocket] = @[]
    updated = false

  with_state_lock:
    if ws in clients:
      var client_info = clients[ws]
      if client_info.room in room_evolution and goal_name in room_evolution[client_info.room].goals:
        var goal = room_evolution[client_info.room].goals[goal_name]
        if goal.phase == "brood" and not goal.evolving and
           (not has_req_gen or req_gen == goal.generation) and
           brood_idx < goal.brood_groups.len and
           cand_idx < goal.brood_groups[brood_idx].candidates_json.len:
          let now = now_ms()
          if not allow_vote_burst(client_info, now):
            clients[ws] = client_info
            throttle_payload = $(%*{
              "type": "vote_throttled",
              "goal": goal_name,
              "index": brood_idx,
              "max_votes_per_user": effective_max_votes(goal, cfg, is_public_image_room(client_info.room, goal)),
              "message": "Vote rate limit reached — pause a moment before clicking again",
            })
          else:
            clients[ws] = client_info
            if ws in goal.brood_groups[brood_idx].votes and goal.brood_groups[brood_idx].votes[ws] == cand_idx:
              goal.brood_groups[brood_idx].votes.del(ws)
            else:
              goal.brood_groups[brood_idx].votes[ws] = cand_idx
            goal.sync_goal_countdown(cfg, now)
            room_evolution[client_info.room].goals[goal_name] = goal
            let r_cnt = if client_info.room in rooms: max(1, rooms[client_info.room].len) else: 1
            votes_payload = build_brood_votes_updated_payload(
              goal, cfg, clients, now, r_cnt, is_public = (client_info.room == "lobby")
            )
            targets = room_targets(client_info.room)
            updated = true

  if throttle_payload.len > 0:
    send_payload(ws, throttle_payload)
  if updated:
    broadcast(targets, votes_payload)

proc handle_leave(ws: WebSocket) {.gcsafe.} =
  var outbox: seq[PendingBroadcast] = @[]
  var evicted = false
  with_state_lock:
    evicted = evict_client_locked(ws, now_ms(), outbox)
  if evicted:
    ws.close()
  for item in outbox:
    broadcast(item.targets, item.payload)

proc on_close(ws: WebSocket) {.gcsafe.} =
  var outbox: seq[PendingBroadcast] = @[]
  with_state_lock:
    discard evict_client_locked(ws, now_ms(), outbox)
  for item in outbox:
    broadcast(item.targets, item.payload)

proc handle_lineage_all(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  let goal_name = sanitize_goal_name(payload.get_or_default("goal").get_str(""))
  var response = ""
  with_state_lock:
    if ws in clients:
      let room = clients[ws].room
      if room in room_evolution and goal_name in room_evolution[room].goals:
        let goal = room_evolution[room].goals[goal_name]
        let lineage_payload = lineage_payload_node(goal, full = true)
        response = $(%*{
          "type": "lineage_all",
          "goal": goal_name,
          "generation": goal.generation,
          "lineage": lineage_payload.lineage,
          "lineage_total": lineage_payload.total,
        })
  if response.len > 0:
    send_payload(ws, response)

proc handle_lineage_active(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  let goal_name = sanitize_goal_name(payload.get_or_default("goal").get_str(""))
  var response = ""
  with_state_lock:
    if ws in clients:
      let room = clients[ws].room
      if room in room_evolution and goal_name in room_evolution[room].goals:
        let goal = room_evolution[room].goals[goal_name]
        let lineage_payload = lineage_payload_node(goal, full = false)
        response = $(%*{
          "type": "lineage_active",
          "goal": goal_name,
          "generation": goal.generation,
          "lineage": lineage_payload.lineage,
          "lineage_total": lineage_payload.total,
        })
  if response.len > 0:
    send_payload(ws, response)

proc collect_genome_slice(
  items: seq[string];
  indices_node: JsonNode;
  max_items: int = 0;
): seq[tuple[idx: int, raw: string]] {.gcsafe.} =
  result = @[]
  let n_items = items.len
  if n_items == 0:
    return
  let cap = if max_items > 0: max_items else: n_items
  if indices_node != nil and indices_node.kind == JArray and indices_node.len > 0:
    var seen = init_hash_set[int]()
    for node in indices_node:
      if node.kind == JInt:
        let idx = node.get_int(-1)
        if idx >= 0 and idx < n_items and idx notin seen:
          seen.incl(idx)
          result.add((idx: idx, raw: items[idx]))
          if result.len >= cap: break
  elif max_items > 0:
    for idx in 0 ..< min(n_items, cap):
      result.add((idx: idx, raw: items[idx]))

proc append_indexed_genome_items(dst: var string; pairs: seq[tuple[idx: int, raw: string]]) {.gcsafe.} =
  dst.add("[")
  for i, p in pairs:
    if i > 0: dst.add(",")
    dst.add("{\"index\":")
    dst.add($p.idx)
    dst.add(",\"genome\":")
    dst.add(p.raw)
    dst.add("}")
  dst.add("]")

proc handle_hydrate_genomes(
  ws: WebSocket;
  payload: JsonNode;
  max_items: int = default_hydrate_max_items;
) {.gcsafe.} =
  let goal_name = sanitize_goal_name(payload.get_or_default("goal").get_str(""))
  if goal_name.len == 0:
    return
  let req_gen = payload.get_or_default("generation").get_int(-1)
  let indices_node = payload.get_or_default("indices")
  var
    selected_pairs: seq[tuple[idx: int, raw: string]] = @[]
    actual_gen = -1
  with_state_lock:
    if ws in clients:
      let room = clients[ws].room
      if room in room_evolution and goal_name in room_evolution[room].goals:
        let goal = room_evolution[room].goals[goal_name]
        if (req_gen < 0 or goal.generation == req_gen) and goal.genome_items.len > 0:
          actual_gen = goal.generation
          selected_pairs = collect_genome_slice(goal.genome_items, indices_node, max_items = max_items)
  if actual_gen < 0 or selected_pairs.len == 0:
    return
  var total_len = goal_name.len + 64
  for pair in selected_pairs: total_len += pair.raw.len + 28
  var resp = new_string_of_cap(total_len)
  resp.add("{\"type\":\"genomes_hydrated\",\"goal\":")
  resp.add(escape_json(goal_name))
  resp.add(",\"generation\":")
  resp.add($actual_gen)
  resp.add(",\"items\":")
  append_indexed_genome_items(resp, selected_pairs)
  resp.add("}")
  send_payload(ws, resp)

proc handle_history_more(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  let before_idx = payload.get_or_default("before").get_int(0)
  let req_limit = clamp(payload.get_or_default("limit").get_int(50), 1, history_limit)
  var response = ""
  with_state_lock:
    if ws in clients:
      let room = clients[ws].room
      var parts: seq[string] = @[]
      if room in room_history:
        parts = room_history[room]
      let total = parts.len
      let end_idx = clamp(before_idx, 0, total)
      let start_idx = max(0, end_idx - req_limit)
      let slice =
        if end_idx > start_idx: parts[start_idx ..< end_idx]
        else: @[]
      let has_more = if start_idx > 0: "true" else: "false"
      response =
        "{\"type\":\"history_more\",\"total\":" & $total & ",\"offset\":" & $start_idx &
        ",\"has_more\":" & has_more & ",\"messages\":[" & slice.join(",") & "]}"
  if response.len > 0:
    send_payload(ws, response)

proc build_evolve_job_and_broadcast(
  room: string;
  goal: var GoalPopulation;
  c: RuntimeConfig;
  cls: Table[WebSocket, ClientInfo];
  targets: seq[WebSocket];
  now: int64;
  outbox: var seq[PendingBroadcast];
): EvoJob {.gcsafe.} =
  goal.evolving = true
  goal.evolving_started_at_ms = now
  goal.countdown_ends_at_ms = 0

  let counted = count_goal_votes(goal)
  let pop_len = goal.votes.len
  var scores = new_seq[int](pop_len)
  var up_counts = new_seq[int](pop_len)
  for i in 0 ..< pop_len:
    scores[i] = counted.stats[i].score
    up_counts[i] = counted.stats[i].up

  if targets.len > 0:
    let r_cnt = max(1, targets.len)
    outbox.add(PendingBroadcast(
      targets: targets,
      payload: build_votes_updated_payload(goal, c, cls, now, r_cnt, is_public = (room == "lobby")),
    ))

  EvoJob(
    kind: EvoJobKind.evolve,
    room: room,
    goal: goal.name,
    task: goal.task,
    generation: goal.generation,
    genomes_json: goal.genomes_json,
    scores: scores,
    up_counts: up_counts,
    population_size: pop_len,
    params: goal.params,
  )

proc handle_set_auto_evolve(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  let goal_name = sanitize_goal_name(payload.get_or_default("goal").get_str(""))
  if goal_name.len == 0:
    return
  let enabled = payload.get_or_default("enabled").get_bool(false)
  let now = now_ms()
  var
    outbox: seq[PendingBroadcast] = @[]
    sys_msg = ""
    targets: seq[WebSocket] = @[]

  with_state_lock:
    if ws in clients:
      let info = clients[ws]
      if info.room in room_evolution and goal_name in room_evolution[info.room].goals:
        var goal = room_evolution[info.room].goals[goal_name]
        let tc = task_cfg_locked(goal.task)
        if tc.sim_phase_enabled and goal.auto_evolve != enabled:
          goal.auto_evolve = enabled
          dirty_rooms.incl(info.room)
          targets = room_targets(info.room)
          let r_cnt = max(1, targets.len)
          if enabled and not goal.evolving and (goal.phase == "voting" or goal.sim_completed >= max(1, goal.votes.len)):
            pending_evo_jobs.add(
              build_evolve_job_and_broadcast(info.room, goal, cfg, clients, targets, now, outbox)
            )
            sys_msg = system_msg_payload(
              info.username & " enabled Auto-Evolve on \"" & goal_name & "\" — advancing generation"
            )
          else:
            if targets.len > 0:
              outbox.add(PendingBroadcast(
                targets: targets,
                payload: build_votes_updated_payload(goal, cfg, clients, now, r_cnt, is_public = (info.room == "lobby")),
              ))
            sys_msg = system_msg_payload(
              info.username & (if enabled: " enabled" else: " disabled") & " Auto-Evolve on \"" & goal_name & "\""
            )
          room_evolution[info.room].goals[goal_name] = goal

  for item in outbox:
    broadcast(item.targets, item.payload)
  if sys_msg.len > 0 and targets.len > 0:
    broadcast(targets, sys_msg)

proc commit_next_generation_locked(
  room: string;
  goal: var GoalPopulation;
  genomes_json: string;
  top_score: int;
  now: int64;
  outbox: var seq[PendingBroadcast];
  sys_msg: var string;
  sys_prefix: string = "Crowd evolved";
) {.gcsafe.} =
  {.gcsafe.}:
    let tc = task_cfg_locked(goal.task)
    let indexed = index_goal_genomes(genomes_json, tc.sim_phase_enabled)
    inc goal.generation
    goal.genomes_json = genomes_json
    goal.sharded_genomes_json = indexed.sharded_json
    goal.genome_items = indexed.items
    goal.uuid_to_idx = indexed.uuid_map
    goal.head_uuid = indexed.head_uuid
    goal.lineage_json = merge_lineage_from_fresh(
      goal.lineage_json,
      extract_lineage_nodes_from_stubs(indexed.stubs_arr, goal.generation),
    )
    goal.refresh_goal_lineage_cache()
    goal.votes = empty_votes(max(1, indexed.items.len))
    goal.clear_goal_brood()
    goal.phase = initial_phase_for_task(tc)
    goal.sim_completed = 0
    goal.countdown_ends_at_ms = 0
    goal.evolving = false
    goal.evolving_started_at_ms = 0
    room_evolution[room].goals[goal.name] = goal
    dirty_rooms.incl(room)
    queue_goal_updated_outbox_locked(room, goal, now, outbox)
    sys_msg = system_msg_payload(
      sys_prefix & " \"" & goal.name & "\" -> gen " & $goal.generation & " (top score " & $top_score & ")"
    )

proc finalize_brood_locked(
  room: string;
  goal: var GoalPopulation;
  now: int64;
  outbox: var seq[PendingBroadcast];
  sys_msg: var string;
  sys_prefix: string = "Brood selection evolved";
) {.gcsafe.} =
  {.gcsafe.}:
    var max_slot = max(0, goal.brood_pop_size - 1)
    for f in goal.brood_fixed:
      if f.slot > max_slot: max_slot = f.slot
    for grp in goal.brood_groups:
      if grp.slot > max_slot: max_slot = grp.slot
    var slots = new_seq[string](max_slot + 1)
    for f in goal.brood_fixed:
      if f.slot >= 0 and f.slot < slots.len and f.genome_json.len > 0:
        slots[f.slot] = f.genome_json
    for grp in goal.brood_groups:
      if grp.slot >= 0 and grp.slot < slots.len and grp.candidates_json.len > 0:
        let win_ci = pick_brood_winner(grp)
        slots[grp.slot] = grp.candidates_json[win_ci]
    var filled: seq[string] = @[]
    for s in slots:
      if s.len > 0:
        filled.add(s)
    if filled.len == 0:
      let tc = task_cfg_locked(goal.task)
      goal.clear_goal_brood()
      goal.phase = initial_phase_for_task(tc)
      goal.countdown_ends_at_ms = 0
      goal.evolving = false
      goal.evolving_started_at_ms = 0
      room_evolution[room].goals[goal.name] = goal
      queue_goal_updated_outbox_locked(room, goal, now, outbox)
      return
    let genomes_json = "[" & filled.join(",") & "]"
    commit_next_generation_locked(
      room, goal, genomes_json, goal.brood_top_score, now, outbox, sys_msg, sys_prefix = sys_prefix
    )

proc handle_evolve_now(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  let goal_name = payload.get_or_default("goal").get_str("")
  if goal_name.len == 0:
    return
  let now = now_ms()
  var
    outbox: seq[PendingBroadcast] = @[]
    denied = false
    sys_msg = ""
    targets: seq[WebSocket] = @[]

  with_state_lock:
    if ws in clients:
      let info = clients[ws]
      if not info.can_evolve_now:
        denied = true
      elif info.room in room_evolution and goal_name in room_evolution[info.room].goals:
        var goal = room_evolution[info.room].goals[goal_name]
        if not goal.evolving:
          targets = room_targets(info.room)
          if goal.phase == "brood" and goal.brood_groups.len > 0:
            finalize_brood_locked(
              info.room, goal, now, outbox, sys_msg,
              sys_prefix = info.username & " finalized brood for",
            )
          else:
            pending_evo_jobs.add(
              build_evolve_job_and_broadcast(info.room, goal, cfg, clients, targets, now, outbox)
            )
            room_evolution[info.room].goals[goal_name] = goal
            sys_msg = system_msg_payload(
              info.username & " evolved \"" & goal_name & "\" early"
            )

  if denied:
    send_payload(ws, $(%*{
      "type": "throttle",
      "message": "Evolve now is restricted to the local/admin session.",
    }))
    return
  for item in outbox:
    broadcast(item.targets, item.payload)
  if sys_msg.len > 0 and targets.len > 0:
    broadcast(targets, sys_msg)

proc handle_select_goal(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  let goal_name = sanitize_goal_name(payload.get_or_default("goal").get_str(""))
  if goal_name.len == 0:
    return
  let need_genomes = payload.get_or_default("need_genomes").get_bool(true)
  var response = ""
  with_state_lock:
    if ws in clients:
      let room = clients[ws].room
      if room in room_evolution and goal_name in room_evolution[room].goals:
        clients[ws].active_goal = goal_name
        if need_genomes:
          let r_cnt = if room in rooms: max(1, rooms[room].len) else: 1
          response = build_goal_updated_payload(
            room_evolution[room].goals[goal_name],
            cfg, clients, now_ms(), r_cnt,
            is_public = (room == "lobby"),
            include_genomes = true,
            defer_adapters = clients[ws].defer_adapters,
            defer_lineage = clients[ws].defer_lineage,
          )
  if response.len > 0:
    send_payload(ws, response)

proc websocket_handler(
  ws: WebSocket;
  event: WebSocketEvent;
  message: Message;
) {.gcsafe.} =
  case event
  of OpenEvent:
    on_open(ws)
  of MessageEvent:
    with_state_lock:
      if ws in clients:
        clients[ws].last_seen_ms = now_ms()
    case message.kind
    of Ping:
      ws.send(message.data, Pong)
      return
    of Pong:
      return
    of BinaryMessage:
      return
    of TextMessage:
      discard
    var payload: JsonNode
    try:
      payload = parse_json(message.data)
    except CatchableError:
      return
    if payload.kind != JObject:
      return
    case payload.get_or_default("type").get_str("chat")
    of "ping": send_payload(ws, """{"type":"pong"}""")
    of "leave": handle_leave(ws)
    of "rename": handle_rename(ws, payload)
    of "cursor": handle_cursor(ws, payload)
    of "chat": handle_chat(ws, payload)
    of "create_goal": handle_create_goal(ws, payload)
    of "select_goal": handle_select_goal(ws, payload)
    of "cast_vote": handle_cast_vote(ws, payload)
    of "cast_brood_vote": handle_cast_brood_vote(ws, payload)
    of "set_auto_evolve": handle_set_auto_evolve(ws, payload)
    of "evolve_now": handle_evolve_now(ws, payload)
    of "lineage_all": handle_lineage_all(ws, payload)
    of "lineage_active": handle_lineage_active(ws, payload)
    of "hydrate_genomes": handle_hydrate_genomes(ws, payload)
    of "history_more": handle_history_more(ws, payload)
    else: discard
  of ErrorEvent:
    discard
  of CloseEvent:
    on_close(ws)

proc ticker_thread_proc(unused: int) {.thread.} =
  {.gcsafe.}:
    while not stop_ticker.load(mo_acquire):
      sleep(ticker_sleep_ms)
      if stop_ticker.load(mo_acquire):
        break

      var outbox: seq[PendingBroadcast] = @[]
      var ping_ws: seq[WebSocket] = @[]
      var stale_ws: seq[WebSocket] = @[]
      let now = now_ms()

      with_state_lock:
        for room, evo in room_evolution.mpairs:
          for g_name in evo.goal_order:
            if g_name notin evo.goals:
              continue
            var goal = evo.goals[g_name]
            if not goal.evolving and goal.countdown_ends_at_ms > 0 and now >= goal.countdown_ends_at_ms:
              let targets = room_targets(room)
              if goal.phase == "brood" and goal.brood_groups.len > 0:
                var b_sys = ""
                finalize_brood_locked(room, goal, now, outbox, b_sys)
                if b_sys.len > 0 and targets.len > 0:
                  outbox.add(PendingBroadcast(targets: targets, payload: b_sys))
              else:
                pending_evo_jobs.add(
                  build_evolve_job_and_broadcast(room, goal, cfg, clients, targets, now, outbox)
                )
                evo.goals[g_name] = goal
            elif goal.evolving and cfg.evolve_watchdog_ms > 0 and goal.evolving_started_at_ms > 0 and
              now - goal.evolving_started_at_ms >= cfg.evolve_watchdog_ms:
              goal.evolving = false
              goal.evolving_started_at_ms = 0
              goal.sync_goal_countdown(cfg, now)
              evo.goals[g_name] = goal
              let targets = room_targets(room)
              if targets.len > 0:
                outbox.add(PendingBroadcast(
                  targets: targets,
                  payload: build_votes_updated_payload(
                    goal, cfg, clients, now, max(1, targets.len), is_public = (room == "lobby")
                  ),
                ))

        for ws, cinfo in clients:
          if cfg.stale_timeout_ms > 0 and cinfo.last_seen_ms > 0 and
            now - cinfo.last_seen_ms >= cfg.stale_timeout_ms:
            stale_ws.add(ws)
          elif cfg.ping_interval_ms > 0 and now - cinfo.last_ping_ms >= cfg.ping_interval_ms:
            ping_ws.add(ws)
        for pws in ping_ws:
          if pws in clients:
            clients[pws].last_ping_ms = now
        for sws in stale_ws:
          discard evict_client_locked(sws, now, outbox)

      for pws in ping_ws:
        pws.send("", Ping)
      for sws in stale_ws:
        sws.close()
      for item in outbox:
        broadcast(item.targets, item.payload)

proc verify_internal_admin(request: Request): tuple[ok: bool, code: int, msg: string] {.gcsafe.} =
  var expected = ""
  with_state_lock:
    expected = admin_password
  if expected.len == 0:
    return (false, 503, "Admin password not configured")
  let supplied = request.headers["X-Admin-Password"]
  if not constant_time_eq(supplied, expected):
    return (false, 403, "Forbidden")
  (true, 200, "")

proc internal_room_goals_handler(request: Request) {.gcsafe.} =
  let auth = verify_internal_admin(request)
  if not auth.ok:
    request.respond(auth.code, body = auth.msg)
    return
  let room = request.path_params["room"]
  if not is_valid_room(room):
    request.respond(400, body = "Invalid room name")
    return
  var obj = new_j_object()
  with_state_lock:
    if room in room_evolution:
      for g_name in room_evolution[room].goal_order:
        if g_name in room_evolution[room].goals:
          obj[g_name] = %room_evolution[room].goals[g_name].votes.len
  request.respond(200, @[("Content-Type", "application/json")], $obj)

proc apply_goal_reset*(room, goal_name, username, genomes_json: string) {.exportpy, gcsafe.}
proc clear_room_evolution*(room, goals_json, goal_name, username: string) {.exportpy, gcsafe.}

proc internal_goal_reset_handler(request: Request) {.gcsafe.} =
  let auth = verify_internal_admin(request)
  if not auth.ok:
    request.respond(auth.code, body = auth.msg)
    return
  var parsed: JsonNode
  try:
    parsed = parse_json(request.body)
  except CatchableError:
    request.respond(400, body = "Invalid JSON")
    return
  let room = parsed.get_or_default("room").get_str("")
  let goal_name = parsed.get_or_default("goal").get_str("")
  let username = parsed.get_or_default("username").get_str("admin")
  let genomes_node = parsed.get_or_default("genomes")
  if room.len == 0 or goal_name.len == 0 or genomes_node == nil or genomes_node.kind != JArray:
    request.respond(400, body = "Missing room, goal, or genomes")
    return
  apply_goal_reset(room, goal_name, username, $genomes_node)
  request.respond(200, @[("Content-Type", "application/json")], """{"ok":true}""")

proc internal_room_clear_handler(request: Request) {.gcsafe.} =
  let auth = verify_internal_admin(request)
  if not auth.ok:
    request.respond(auth.code, body = auth.msg)
    return
  var parsed: JsonNode
  try:
    parsed = parse_json(request.body)
  except CatchableError:
    request.respond(400, body = "Invalid JSON")
    return
  let room = parsed.get_or_default("room").get_str("")
  let goal_name = parsed.get_or_default("goal").get_str("")
  let username = parsed.get_or_default("username").get_str("admin")
  let goals_node = parsed.get_or_default("goals")
  if room.len == 0 or goals_node == nil or goals_node.kind != JArray:
    request.respond(400, body = "Missing room or goals")
    return
  clear_room_evolution(room, $goals_node, goal_name, username)
  request.respond(200, @[("Content-Type", "application/json")], """{"ok":true}""")

proc build_room_snapshot_json(
  room: string;
  filter_goal: string = "";
  include_lineage: bool = false;
  stubs_only: bool = false;
  include_genomes: bool = true;
): string {.gcsafe.} =
  var parts: seq[string] = @[]
  var total_len = room.len + 36
  with_state_lock:
    if room in room_evolution and not room_evolution[room].initializing:
      let evo = room_evolution[room]
      for g_name in evo.goal_order:
        if filter_goal.len > 0 and g_name != filter_goal:
          continue
        if g_name in evo.goals:
          let g = evo.goals[g_name]
          let g_genomes =
            if not include_genomes: "[]"
            elif stubs_only and g.sharded_genomes_json.len > 0: g.sharded_genomes_json
            elif g.genomes_json.len > 0: g.genomes_json
            else: "[]"
          let env_p = if g.env_params_json.len > 0: g.env_params_json else: "{}"
          let head_u = g.head_uuid
          let l_str =
            if include_lineage and g.lineage_json.len > 0: g.lineage_json
            else: ""
          let task_str = if g.task.len > 0: g.task else: "image"
          var item = new_string_of_cap(g_genomes.len + env_p.len + l_str.len + 184)
          item.add("{\"name\":")
          item.add(escape_json(g.name))
          item.add(",\"task\":")
          item.add(escape_json(task_str))
          item.add(",\"env_params\":")
          item.add(env_p)
          item.add(",\"generation\":")
          item.add($g.generation)
          item.add(",\"population_size\":")
          item.add($g.votes.len)
          item.add(",\"head_uuid\":")
          item.add(escape_json(head_u))
          item.add(",\"phase\":")
          item.add(escape_json(if g.evolving: "evolving" elif g.phase.len > 0: g.phase else: "voting"))
          item.add(",\"sim_completed\":")
          item.add($g.sim_completed)
          item.add(",\"auto_evolve\":")
          item.add(if g.auto_evolve: "true" else: "false")
          item.add(",\"brood_size\":")
          item.add($g.params.brood_size)
          item.add(",\"genomes\":")
          item.add(g_genomes)
          if include_lineage:
            item.add(",\"lineage\":")
            item.add(if l_str.len > 0: l_str else: "[]")
          item.add("}")
          total_len += item.len + 1
          parts.add(item)
  result = new_string_of_cap(total_len)
  result.add("{\"room\":")
  result.add(escape_json(room))
  result.add(",\"goals\":[")
  for idx, p in parts:
    if idx > 0:
      result.add(",")
    result.add(p)
  result.add("]}")

proc get_room_snapshot_json*(room: string): string {.exportpy, gcsafe.} =
  ## Return current active goal states (name, task, envParams, generation, genomes) for `room`.
  build_room_snapshot_json(room)

proc get_room_snapshot_with_lineage_json*(room: string): string {.exportpy, gcsafe.} =
  ## Return current active goal states including full ancestor lineage for individual UUID lookup.
  build_room_snapshot_json(room, "", include_lineage = true)

proc get_active_rooms_json*(): string {.exportpy, gcsafe.} =
  ## Return JSON array of room names currently loaded in memory.
  var arr = new_j_array()
  with_state_lock:
    for r in room_evolution.keys:
      if not room_evolution[r].initializing:
        arr.add(%r)
  $arr

proc get_room_goal_snapshot_json*(room, goal_name: string): string {.exportpy, gcsafe.} =
  ## Return snapshot for a single goal in `room` without serializing unrelated tabs.
  build_room_snapshot_json(room, sanitize_goal_name(goal_name))

proc get_room_goal_stubs_json*(room, goal_name: string): string {.exportpy, gcsafe.} =
  ## Return lightweight metadata-only genome stubs (omitting LoRA adapters) for shard scheduling & rankings.
  build_room_snapshot_json(room, sanitize_goal_name(goal_name), stubs_only = true)

proc get_room_goal_head_json*(room, goal_name: string): string {.exportpy, gcsafe.} =
  ## Return O(1) ~200-byte goal header (generation, populationSize, phase, envParams) with zero genomes serialized.
  build_room_snapshot_json(room, sanitize_goal_name(goal_name), include_genomes = false)

proc get_room_goal_slice_json*(room, goal_name: string; generation: int; indices_json: string): string {.exportpy, gcsafe.} =
  ## Return full genome payloads ONLY for requested `indices_json` in O(batch_size) without reparsing all P genomes.
  let clean_goal = sanitize_goal_name(goal_name)
  var req_indices: JsonNode
  try:
    req_indices = parse_json(indices_json)
  except CatchableError:
    return """{"ok":false,"items":[]}"""
  if req_indices.kind != JArray:
    return """{"ok":false,"items":[]}"""
  var pairs: seq[tuple[idx: int, raw: string]] = @[]
  var actual_gen = -1
  with_state_lock:
    if room in room_evolution and not room_evolution[room].initializing and clean_goal in room_evolution[room].goals:
      let g = room_evolution[room].goals[clean_goal]
      if generation < 0 or g.generation == generation:
        actual_gen = g.generation
        pairs = collect_genome_slice(g.genome_items, req_indices)
  if actual_gen < 0:
    return """{"ok":false,"items":[]}"""
  var total_len = 48
  for p in pairs: total_len += p.raw.len + 28
  result = new_string_of_cap(total_len)
  result.add("{\"ok\":true,\"generation\":")
  result.add($actual_gen)
  result.add(",\"items\":")
  append_indexed_genome_items(result, pairs)
  result.add("}")

proc pack_uuid_hit_json(room, goal, task, env_json: string; gen, idx: int; is_live: bool; genome_raw: string): string {.gcsafe.} =
  result = new_string_of_cap(genome_raw.len + env_json.len + 160)
  result.add(if is_live: "{\"ok\":true,\"is_live\":true,\"room\":" else: "{\"ok\":true,\"is_live\":false,\"room\":")
  result.add(escape_json(room))
  result.add(",\"goal\":")
  result.add(escape_json(goal))
  result.add(",\"task\":")
  result.add(escape_json(task))
  result.add(",\"env_params\":")
  result.add(env_json)
  result.add(",\"generation\":")
  result.add($gen)
  result.add(",\"index\":")
  result.add(if is_live: $idx else: "null")
  result.add(",\"genome\":")
  result.add(genome_raw)
  result.add("}")

proc get_room_individual_by_uuid_json*(room, ind_uuid: string): string {.exportpy, gcsafe.} =
  ## O(1) lookup of a live individual by 8-char UUID in `room` (falling back to room lineage only on miss).
  let clean_u = ind_uuid.strip().to_lower_ascii()
  if clean_u.len != 8:
    return ""
  var
    hit_goal = ""
    hit_task = "image"
    hit_env = "{}"
    hit_gen = 0
    hit_idx = -1
    hit_raw = ""
    lineage_candidates: seq[tuple[g_name, task_str, env_str, lin_json: string]] = @[]
  with_state_lock:
    if room in room_evolution and not room_evolution[room].initializing:
      let evo = room_evolution[room]
      for g_name in evo.goal_order:
        if g_name notin evo.goals: continue
        let g = evo.goals[g_name]
        let t_str = if g.task.len > 0: g.task else: "image"
        let e_str = if g.env_params_json.len > 0: g.env_params_json else: "{}"
        if clean_u in g.uuid_to_idx:
          let idx = g.uuid_to_idx[clean_u]
          if idx >= 0 and idx < g.genome_items.len:
            hit_goal = g.name; hit_task = t_str; hit_env = e_str
            hit_gen = g.generation; hit_idx = idx; hit_raw = g.genome_items[idx]
            break
        if g.lineage_json.len > 0:
          lineage_candidates.add((g.name, t_str, e_str, g.lineage_json))
  if hit_raw.len > 0:
    return pack_uuid_hit_json(room, hit_goal, hit_task, hit_env, hit_gen, hit_idx, true, hit_raw)
  for cand in lineage_candidates:
    try:
      let parsed = parse_json(cand.lin_json)
      if parsed.kind == JArray:
        for node in parsed:
          if node.kind == JObject and node.get_or_default("uuid").get_str("").strip().to_lower_ascii() == clean_u:
            return pack_uuid_hit_json(room, cand.g_name, cand.task_str, cand.env_str, node.get_or_default("gen").get_int(0), -1, false, $node)
    except CatchableError:
      discard
  ""

proc broadcast_goal_payload*(room, goal_name, payload_json: string) {.exportpy, gcsafe.} =
  if room.len == 0 or payload_json.len == 0:
    return
  let target_goal = sanitize_goal_name(goal_name)
  var targets: seq[WebSocket] = @[]
  with_state_lock:
    if room in rooms:
      for ws in rooms[room]:
        if target_goal.len > 0 and ws in clients and clients[ws].lazy_goals and clients[ws].active_goal != target_goal:
          continue
        targets.add(ws)
  if targets.len > 0:
    broadcast(targets, payload_json)

proc broadcast_room_payload*(room, payload_json: string) {.exportpy, gcsafe.} =
  broadcast_goal_payload(room, "", payload_json)

proc internal_room_snapshot_handler(request: Request) {.gcsafe.} =
  let auth = verify_internal_admin(request)
  if not auth.ok:
    request.respond(auth.code, body = auth.msg)
    return
  let room = request.path_params["room"]
  if not is_valid_room(room):
    request.respond(400, body = "Invalid room name")
    return
  let goal_filter = sanitize_goal_name(request.query_params["goal"])
  let stubs_only = request.query_params["stubs"] == "1"
  let head_only = request.query_params["head"] == "1"
  request.respond(
    200,
    @[("Content-Type", "application/json")],
    build_room_snapshot_json(room, goal_filter, stubs_only = stubs_only, include_genomes = not head_only),
  )

proc internal_room_slice_handler(request: Request) {.gcsafe.} =
  let auth = verify_internal_admin(request)
  if not auth.ok:
    request.respond(auth.code, body = auth.msg)
    return
  var parsed: JsonNode
  try:
    parsed = parse_json(request.body)
  except CatchableError:
    request.respond(400, body = "Invalid JSON")
    return
  let room = parsed.get_or_default("room").get_str("")
  let goal_name = parsed.get_or_default("goal").get_str("")
  let generation = parsed.get_or_default("generation").get_int(-1)
  let indices_node = parsed.get_or_default("indices")
  if not is_valid_room(room) or goal_name.len == 0 or indices_node == nil or indices_node.kind != JArray:
    request.respond(400, body = "Missing room, goal, or indices")
    return
  request.respond(200, @[("Content-Type", "application/json")], get_room_goal_slice_json(room, goal_name, generation, $indices_node))

proc internal_room_broadcast_handler(request: Request) {.gcsafe.} =
  let auth = verify_internal_admin(request)
  if not auth.ok:
    request.respond(auth.code, body = auth.msg)
    return
  var parsed: JsonNode
  try:
    parsed = parse_json(request.body)
  except CatchableError:
    request.respond(400, body = "Invalid JSON")
    return
  let room = parsed.get_or_default("room").get_str("")
  let goal_filter = sanitize_goal_name(parsed.get_or_default("goal").get_str(""))
  let payload_node = parsed.get_or_default("payload")
  if room.len == 0 or payload_node == nil or payload_node.kind != JObject:
    request.respond(400, body = "Missing room or payload")
    return
  broadcast_goal_payload(room, goal_filter, $payload_node)
  request.respond(200, @[("Content-Type", "application/json")], """{"ok":true}""")

proc get_room_peer_count*(room: string): int {.exportpy, gcsafe.} =
  ## Return number of unique active client sessions currently connected to `room`.
  with_state_lock:
    if room in rooms:
      var uniq_cids = init_hash_set[string]()
      var anon_cnt = 0
      for ws in rooms[room]:
        if ws in clients:
          let c = clients[ws].cid
          if c.len >= 8:
            uniq_cids.incl(c)
          else:
            inc anon_cnt
      result = uniq_cids.len + anon_cnt

proc notify_goal_sim_progress*(
  room, goal_name: string;
  generation, verified_count: int;
): string {.exportpy, gcsafe.} =
  ## Update crowdsourced simulation progress and advance phase (`simulating` -> `voting` | `evolving`).
  var
    outbox: seq[PendingBroadcast] = @[]
    sys_msg = ""
    targets: seq[WebSocket] = @[]
    next_phase = ""

  with_state_lock:
    if room in room_evolution and goal_name in room_evolution[room].goals:
      let now = now_ms()
      var goal = room_evolution[room].goals[goal_name]
      if goal.generation == generation and not goal.evolving:
        let pop_len = max(1, goal.votes.len)
        let clamped_v = clamp(verified_count, 0, pop_len)
        let prev_v = goal.sim_completed
        if clamped_v > goal.sim_completed:
          goal.sim_completed = clamped_v
        targets = room_targets(room)
        let r_cnt = max(1, targets.len)
        if goal.phase == "simulating" and goal.sim_completed >= pop_len:
          dirty_rooms.incl(room)
          if goal.auto_evolve:
            pending_evo_jobs.add(
              build_evolve_job_and_broadcast(room, goal, cfg, clients, targets, now, outbox)
            )
            next_phase = "evolving"
            sys_msg = system_msg_payload(
              "Simulations complete (" & $pop_len & "/" & $pop_len & ") — auto-evolving \"" & goal_name & "\""
            )
          else:
            goal.phase = "voting"
            goal.sync_goal_countdown(cfg, now)
            next_phase = "voting"
            if targets.len > 0:
              outbox.add(PendingBroadcast(
                targets: targets,
                payload: build_votes_updated_payload(goal, cfg, clients, now, r_cnt, is_public = (room == "lobby")),
              ))
            sys_msg = system_msg_payload(
              "Simulations complete for \"" & goal_name & "\" gen " & $generation & " (" & $pop_len & "/" & $pop_len & ") — voting open"
            )
        elif goal.sim_completed != prev_v:
          next_phase = goal.phase
          if targets.len > 0:
            outbox.add(PendingBroadcast(
              targets: targets,
              payload: build_votes_updated_payload(goal, cfg, clients, now, r_cnt, is_public = (room == "lobby")),
            ))
        else:
          next_phase = goal.phase
        room_evolution[room].goals[goal_name] = goal
      elif goal.evolving:
        next_phase = "evolving"
      else:
        next_phase = goal.phase

  for item in outbox:
    broadcast(item.targets, item.payload)
  if sys_msg.len > 0 and targets.len > 0:
    broadcast(targets, sys_msg)
  next_phase

proc set_goal_auto_evolve*(room, goal_name: string; enabled: bool): string {.exportpy, gcsafe.} =
  var
    outbox: seq[PendingBroadcast] = @[]
    targets: seq[WebSocket] = @[]
    res_phase = ""
  with_state_lock:
    if room in room_evolution and goal_name in room_evolution[room].goals:
      let now = now_ms()
      var goal = room_evolution[room].goals[goal_name]
      goal.auto_evolve = enabled
      dirty_rooms.incl(room)
      targets = room_targets(room)
      let r_cnt = max(1, targets.len)
      if enabled and not goal.evolving and (goal.phase == "voting" or goal.sim_completed >= max(1, goal.votes.len)):
        pending_evo_jobs.add(
          build_evolve_job_and_broadcast(room, goal, cfg, clients, targets, now, outbox)
        )
        res_phase = "evolving"
      else:
        res_phase = if goal.evolving: "evolving" else: goal.phase
        if targets.len > 0:
          outbox.add(PendingBroadcast(
            targets: targets,
            payload: build_votes_updated_payload(goal, cfg, clients, now, r_cnt, is_public = (room == "lobby")),
          ))
      room_evolution[room].goals[goal_name] = goal
  for item in outbox:
    broadcast(item.targets, item.payload)
  res_phase

proc internal_room_peers_handler(request: Request) {.gcsafe.} =
  let auth = verify_internal_admin(request)
  if not auth.ok:
    request.respond(auth.code, body = auth.msg)
    return
  let room = request.path_params["room"]
  if not is_valid_room(room):
    request.respond(400, body = "Invalid room name")
    return
  let cnt = get_room_peer_count(room)
  request.respond(200, @[("Content-Type", "application/json")], $(%*{"ok": true, "peers": cnt}))

proc internal_sim_progress_handler(request: Request) {.gcsafe.} =
  let auth = verify_internal_admin(request)
  if not auth.ok:
    request.respond(auth.code, body = auth.msg)
    return
  var parsed: JsonNode
  try:
    parsed = parse_json(request.body)
  except CatchableError:
    request.respond(400, body = "Invalid JSON")
    return
  let room = parsed.get_or_default("room").get_str("")
  let goal_name = parsed.get_or_default("goal").get_str("")
  let generation = parsed.get_or_default("generation").get_int(0)
  let verified_count = parsed.get_or_default("verified_count").get_int(0)
  let phase = notify_goal_sim_progress(room, goal_name, generation, verified_count)
  request.respond(200, @[("Content-Type", "application/json")], $(%*{"ok": true, "phase": phase}))

proc internal_room_uuid_handler(request: Request) {.gcsafe.} =
  let auth = verify_internal_admin(request)
  if not auth.ok:
    request.respond(auth.code, body = auth.msg)
    return
  let room = request.path_params["room"]
  let ind_uuid = request.path_params["uuid"]
  if not is_valid_room(room):
    request.respond(400, body = "Invalid room name")
    return
  let found = get_room_individual_by_uuid_json(room, ind_uuid)
  if found.len == 0:
    request.respond(404, @[("Content-Type", "application/json")], """{"ok":false}""")
    return
  request.respond(200, @[("Content-Type", "application/json")], found)

proc ws_health_handler(request: Request) {.gcsafe.} =
  request.respond(200, @[("Content-Type", "application/json")], """{"status":"ok","service":"ws"}""")

proc serve_thread_proc(args: tuple[host: string, port: int]) {.thread.} =
  {.gcsafe.}:
    var srv: Server = nil
    try:
      var router: Router
      router.get("/health", ws_health_handler)
      router.get("/health/", ws_health_handler)
      router.get("/ws/chat/@room", upgrade_handler)
      router.get("/ws/chat/@room/", upgrade_handler)
      router.get("/internal/room-goals/@room", internal_room_goals_handler)
      router.get("/internal/room-snapshot/@room", internal_room_snapshot_handler)
      router.post("/internal/room-slice", internal_room_slice_handler)
      router.get("/internal/room-uuid/@room/@uuid", internal_room_uuid_handler)
      router.get("/internal/room-peers/@room", internal_room_peers_handler)
      router.post("/internal/room-broadcast", internal_room_broadcast_handler)
      router.post("/internal/sim-progress", internal_sim_progress_handler)
      router.post("/internal/goal-reset", internal_goal_reset_handler)
      router.post("/internal/room-clear", internal_room_clear_handler)
      srv = new_server(router, websocket_handler, log_handler = quiet_logger)
      with_lock ready_lock:
        server_ptr = srv
        server_ready.store(true, mo_release)
      signal(ready_cond)
      srv.serve(Port(args.port), args.host)
    except CatchableError:
      with_lock ready_lock:
        server_failed.store(true, mo_release)
        server_ready.store(true, mo_release)
      signal(ready_cond)

# --- Python exports ---

proc start_server*(host: string; port: int) {.exportpy.} =
  var srv: Server = nil
  with_state_lock:
    if is_running:
      return
    server_ready.store(false, mo_release)
    server_failed.store(false, mo_release)
    stop_ticker.store(false, mo_release)
    server_ptr = nil
    is_running = true
    create_thread(server_thread, serve_thread_proc, (host, port))

  with_lock ready_lock:
    while not server_ready.load(mo_acquire):
      wait(ready_cond, ready_lock)
    srv = server_ptr

  if server_failed.load(mo_acquire) or srv == nil:
    with_state_lock:
      is_running = false
    join_thread(server_thread)
    raise new_exception(CatchableError, "Failed to start Mummy WebSocket server")

  srv.wait_until_ready()
  create_thread(ticker_thread, ticker_thread_proc, 0)

proc stop_server*() {.exportpy.} =
  var srv: Server = nil
  var should_join = false
  with_state_lock:
    if is_running and server_ptr != nil:
      srv = server_ptr
      is_running = false
      should_join = true
  if should_join:
    stop_ticker.store(true, mo_release)
    join_thread(ticker_thread)
  if srv != nil:
    srv.close()
  if should_join:
    join_thread(server_thread)
  with_state_lock:
    server_ptr = nil
    clients.clear()
    pending_upgrades.clear()
    next_upgrade_seq = 0
    rooms.clear()

proc set_round_config*(min_votes, countdown_ms: int; override_min_votes: bool = false) {.exportpy.} =
  with_state_lock:
    if min_votes > 0:
      cfg.min_votes = clamp(min_votes, 1, max_round_min_votes)
      cfg.min_votes_explicit = true
      cfg.min_votes_override = override_min_votes
    else:
      cfg.min_votes_override = false
    cfg.countdown_ms = max(50'i64, int64(countdown_ms))

proc set_heartbeat_config*(ping_interval_ms, stale_timeout_ms: int; evolve_watchdog_ms: int = 30_000) {.exportpy.} =
  with_state_lock:
    cfg.ping_interval_ms = max(0'i64, int64(ping_interval_ms))
    cfg.stale_timeout_ms = max(0'i64, int64(stale_timeout_ms))
    cfg.evolve_watchdog_ms = max(0'i64, int64(evolve_watchdog_ms))

proc set_default_population_size*(pop_size: int) {.exportpy.} =
  with_state_lock:
    cfg.population_size = clamp(pop_size, min_population_size, max_population_size)

proc set_default_max_votes*(max_votes: int) {.exportpy.} =
  with_state_lock:
    cfg.params.max_votes_per_user = clamp(max_votes, min_votes_per_user, max_votes_per_user)

proc set_default_ga_params*(mut_sigma, elite_frac, crossover_rate: float) {.exportpy.} =
  with_state_lock:
    cfg.params.mut_sigma = clamp(mut_sigma, min_mut_sigma, max_mut_sigma)
    cfg.params.elite_frac = clamp(elite_frac, min_elite_frac, max_elite_frac)
    cfg.params.crossover_rate = clamp(crossover_rate, min_crossover_rate, max_crossover_rate)

proc set_default_brood_size*(brood_size: int) {.exportpy.} =
  with_state_lock:
    cfg.params.brood_size = clamp(brood_size, min_brood_size, max_brood_size)

proc register_task_configs*(configs_json: string) {.exportpy.} =
  let parsed = parse_json(configs_json)
  if parsed.kind != JArray:
    return
  with_state_lock:
    for item in parsed:
      if item.kind != JObject:
        continue
      let tid = item.get_or_default("task").get_str("").strip().to_lower_ascii()
      if tid.len == 0 or tid.len > max_task_name_len:
        continue
      let has_elite = item.has_key("default_elite_frac") and item["default_elite_frac"].kind in {JFloat, JInt}
      task_configs[tid] = TaskConfig(
        is_control: item.get_or_default("is_control").get_bool(false),
        sim_phase_enabled: item.get_or_default("sim_phase_enabled").get_bool(false),
        default_auto_evolve: item.get_or_default("default_auto_evolve").get_bool(false),
        default_pop_size: clamp(
          item.get_or_default("default_pop_size").get_int(default_control_population_size),
          min_population_size,
          max_population_size,
        ),
        max_pop_size: clamp(
          item.get_or_default("max_pop_size").get_int(control_max_population_size),
          min_population_size,
          max_population_size,
        ),
        default_min_votes: max(0, min(max_round_min_votes, item.get_or_default("default_min_votes").get_int(0))),
        crowd_shard_threshold: clamp(
          item.get_or_default("crowd_shard_threshold").get_int(default_crowd_shard_threshold),
          min_population_size,
          1024,
        ),
        has_elite_override: has_elite,
        default_elite_frac: (
          if has_elite: clamp(item["default_elite_frac"].get_float(0.0), min_elite_frac, max_elite_frac)
          else: default_elite_frac
        ),
        default_brood_size: clamp(
          item.get_or_default("default_brood_size").get_int(default_brood_size),
          min_brood_size,
          max_brood_size,
        ),
      )

proc seed_room_goals*(room, goals_json: string) {.exportpy.} =
  ## Seed a room with initial PyTorch-generated goals JSON: [{"name": ..., "genomes": [...]}]
  let det = json_detach_goals_array(goals_json)
  let parsed = parse_json(det.lean_arr)
  if parsed.kind != JArray:
    return
  with_state_lock:
    room_evolution[room] = populate_room_goals_locked(parsed, cfg.params, det.genomes_list)
    dirty_rooms.incl(room)

proc drain_evo_jobs*(): string {.exportpy.} =
  var jobs: seq[EvoJob]
  with_state_lock:
    if pending_evo_jobs.len == 0:
      return "[]"
    jobs = move pending_evo_jobs
    pending_evo_jobs = @[]
  var parts: seq[string] = @[]
  for job in jobs:
    let meta = %*{
      "kind": $job.kind,
      "room": job.room,
      "goal": job.goal,
      "task": (if job.task.len > 0: job.task else: "image"),
      "generation": job.generation,
      "username": job.username,
      "scores": job.scores,
      "up_counts": job.up_counts,
      "mut_sigma": job.params.mut_sigma,
      "elite_frac": job.params.elite_frac,
      "crossover_rate": job.params.crossover_rate,
      "brood_size": job.params.brood_size,
      "population_size": job.population_size,
      "max_votes_per_user": job.params.max_votes_per_user,
      "min_votes": job.params.min_votes,
    }
    let meta_str = $meta
    let g_str = if job.genomes_json.len > 0: job.genomes_json else: "[]"
    let ep_str = if job.env_params_json.len > 0: job.env_params_json else: "{}"
    var item = new_string_of_cap(meta_str.len + g_str.len + ep_str.len + 32)
    item.add(meta_str[0 ..< meta_str.len - 1])
    item.add(",\"env_params\":")
    item.add(ep_str)
    item.add(",\"genomes\":")
    item.add(g_str)
    item.add("}")
    parts.add(item)
  "[" & parts.join(",") & "]"

proc apply_room_init*(room, goals_json: string) {.exportpy.} =
  let det = json_detach_goals_array(goals_json)
  let parsed = parse_json(det.lean_arr)
  if parsed.kind != JArray:
    return
  var outbox: seq[PendingBroadcast] = @[]
  with_state_lock:
    let dp =
      if room in room_evolution and room_evolution[room].default_params.max_votes_per_user > 0:
        room_evolution[room].default_params
      else:
        cfg.params
    var carry_creates = init_hash_set[string]()
    var carry_tasks = init_table[string, string]()
    var carry_params = init_table[string, GoalParams]()
    var carry_env_params = init_table[string, string]()
    if room in room_evolution and room_evolution[room].initializing:
      # keep goals requested before the room finished booting
      carry_creates = room_evolution[room].pending_creates
      carry_tasks = room_evolution[room].pending_create_tasks
      carry_params = room_evolution[room].pending_create_params
      carry_env_params = room_evolution[room].pending_create_env_params
    room_evolution[room] = populate_room_goals_locked(parsed, dp, det.genomes_list)
    room_evolution[room].pending_creates = carry_creates
    room_evolution[room].pending_create_tasks = carry_tasks
    room_evolution[room].pending_create_params = carry_params
    room_evolution[room].pending_create_env_params = carry_env_params
    dirty_rooms.incl(room)
    queue_evolution_state_outbox_locked(room, outbox)
  for item in outbox:
    broadcast(item.targets, item.payload)

proc apply_goal_create_impl(
  room, goal_name, task_override, username, genomes_json: string;
  min_votes: int = -1;
  env_params_override: string = "";
) =
  var
    outbox: seq[PendingBroadcast] = @[]
    sys_payload: string
    targets: seq[WebSocket]
    applied = false
  with_state_lock:
    if room in room_evolution:
      room_evolution[room].pending_creates.excl(goal_name)
      var task_name = if task_override.len > 0: sanitize_task_name_locked(task_override) else: "image"
      if goal_name in room_evolution[room].pending_create_tasks:
        if task_override.len == 0:
          task_name = sanitize_task_name_locked(room_evolution[room].pending_create_tasks[goal_name])
        room_evolution[room].pending_create_tasks.del(goal_name)
      var env_p_json = if env_params_override.len > 0: env_params_override else: "{}"
      if goal_name in room_evolution[room].pending_create_env_params:
        if env_params_override.len == 0:
          env_p_json = room_evolution[room].pending_create_env_params[goal_name]
        room_evolution[room].pending_create_env_params.del(goal_name)
      var gp = room_evolution[room].default_params
      if gp.max_votes_per_user <= 0:
        gp = cfg.params
      if goal_name in room_evolution[room].pending_create_params:
        gp = room_evolution[room].pending_create_params[goal_name]
        room_evolution[room].pending_create_params.del(goal_name)
      if min_votes > 0:
        gp.min_votes = clamp(min_votes, 1, max_round_min_votes)
      elif gp.min_votes <= 0:
        gp.min_votes = default_min_votes_for_task_locked(task_name)
      if goal_name notin room_evolution[room].goals:
        room_evolution[room].goal_order.add(goal_name)
        room_evolution[room].goals[goal_name] = init_goal_population_locked(
          goal_name, task_name, genomes_json, 0, gp, env_params_json = env_p_json
        )
        dirty_rooms.incl(room)
        queue_evolution_state_outbox_locked(room, outbox)
        sys_payload = system_msg_payload(username & " created goal \"" & goal_name & "\"")
        targets = room_targets(room)
        applied = true
  if applied:
    for item in outbox:
      broadcast(item.targets, item.payload)
    if targets.len > 0:
      broadcast(targets, sys_payload)

proc apply_goal_create*(room, goal_name, username, genomes_json: string) {.exportpy.} =
  apply_goal_create_impl(room, goal_name, "", username, genomes_json)

proc apply_goal_create_with_env*(room, goal_name, username, genomes_json, env_params_json: string) {.exportpy.} =
  apply_goal_create_impl(room, goal_name, "", username, genomes_json, -1, env_params_json)

proc apply_goal_create_with_task*(room, goal_name, task_name, username, genomes_json: string; min_votes: int = -1) {.exportpy.} =
  apply_goal_create_impl(room, goal_name, task_name, username, genomes_json, min_votes)

proc clear_room_evolution*(room, goals_json, goal_name, username: string) {.exportpy, gcsafe.} =
  let target_goal = goal_name.strip()
  let single_goal_mode = target_goal.len > 0 and target_goal != "*"
  var
    parsed: JsonNode = new_j_array()
    detached_genomes: seq[string] = @[]
    single_items_len = 0
  if single_goal_mode:
    let trimmed = goals_json.strip()
    if trimmed.len == 0 or not trimmed.starts_with("["):
      return
    single_items_len = json_split_array_items(trimmed).len
  else:
    let det = json_detach_goals_array(goals_json)
    try:
      parsed = parse_json(det.lean_arr)
    except CatchableError:
      return
    if parsed.kind != JArray:
      return
    detached_genomes = det.genomes_list
  var
    outbox: seq[PendingBroadcast] = @[]
    sys_payload = ""
    targets: seq[WebSocket] = @[]
  with_state_lock:
    let dp =
      if room in room_evolution and room_evolution[room].default_params.max_votes_per_user > 0:
        room_evolution[room].default_params
      else:
        cfg.params
    if single_goal_mode:
      if room in room_evolution:
        var evo = room_evolution[room]
        evo.pending_creates.excl(target_goal)
        evo.pending_create_tasks.del(target_goal)
        evo.pending_create_params.del(target_goal)
        evo.pending_create_env_params.del(target_goal)
        if single_items_len == 0:
          evo.goals.del(target_goal)
          var next_order: seq[string] = @[]
          for gn in evo.goal_order:
            if gn != target_goal:
              next_order.add(gn)
          evo.goal_order = next_order
          room_evolution[room] = evo
          if room != "lobby" and evo.goal_order.len == 0 and rooms.get_or_default(room).len == 0:
            room_evolution.del(room)
            dirty_rooms.excl(room)
          else:
            dirty_rooms.incl(room)
            queue_evolution_state_outbox_locked(room, outbox)
            sys_payload = system_msg_payload(username & " cleared \"" & target_goal & "\"")
            targets = room_targets(room)
        else:
          var gp = dp
          var goal_task = "image"
          var goal_env_json = "{}"
          if target_goal in evo.goals:
            gp = evo.goals[target_goal].params
            goal_task = evo.goals[target_goal].task
            goal_env_json = evo.goals[target_goal].env_params_json
          else:
            evo.goal_order.add(target_goal)
          evo.goals[target_goal] = init_goal_population_locked(
            target_goal, goal_task, goals_json, single_items_len, gp, env_params_json = goal_env_json
          )
          room_evolution[room] = evo
          dirty_rooms.incl(room)
          queue_goal_updated_outbox_locked(room, evo.goals[target_goal], now_ms(), outbox)
          sys_payload = system_msg_payload(username & " cleared \"" & target_goal & "\" to gen 0")
          targets = room_targets(room)
    else:
      if room != "lobby" and parsed.len == 0:
        dirty_rooms.excl(room)
        if rooms.get_or_default(room).len == 0:
          room_evolution.del(room)
        else:
          room_evolution[room] = RoomEvolution(
            goal_order: @[],
            goals: init_table[string, GoalPopulation](),
            pending_creates: init_hash_set[string](),
            pending_create_tasks: init_table[string, string](),
            pending_create_params: init_table[string, GoalParams](),
            pending_create_env_params: init_table[string, string](),
            default_params: dp,
            initializing: false,
          )
          queue_evolution_state_outbox_locked(room, outbox)
          sys_payload = system_msg_payload(username & " cleared room evolution state")
          targets = room_targets(room)
      else:
        room_evolution[room] = populate_room_goals_locked(parsed, dp, detached_genomes)
        dirty_rooms.incl(room)
        queue_evolution_state_outbox_locked(room, outbox)
        sys_payload = system_msg_payload(username & " cleared room evolution state to gen 0")
        targets = room_targets(room)
  for item in outbox:
    broadcast(item.targets, item.payload)
  if targets.len > 0 and sys_payload.len > 0:
    broadcast(targets, sys_payload)

proc apply_goal_reset*(room, goal_name, username, genomes_json: string) {.exportpy.} =
  var
    outbox: seq[PendingBroadcast] = @[]
    sys_payload: string
    targets: seq[WebSocket]
    applied = false
  with_state_lock:
    if room in room_evolution and goal_name in room_evolution[room].goals:
      let now = now_ms()
      var goal = room_evolution[room].goals[goal_name]
      let tc = task_cfg_locked(goal.task)
      let indexed = index_goal_genomes(genomes_json, tc.sim_phase_enabled)
      let fresh_lineage = merge_lineage_from_fresh("", extract_lineage_nodes_from_stubs(indexed.stubs_arr, 0))
      let fresh_lean = compute_lean_lineage(fresh_lineage, genomes_json, 0)
      goal.generation = 0
      goal.genomes_json = genomes_json
      goal.sharded_genomes_json = indexed.sharded_json
      goal.genome_items = indexed.items
      goal.uuid_to_idx = indexed.uuid_map
      goal.head_uuid = indexed.head_uuid
      goal.lineage_json = fresh_lineage
      goal.lean_lineage_json = fresh_lean.lean_json
      goal.lineage_total = fresh_lean.total
      goal.votes = empty_votes(max(1, indexed.items.len))
      goal.clear_goal_brood()
      goal.phase = initial_phase_for_task(tc)
      goal.sim_completed = 0
      goal.countdown_ends_at_ms = 0
      goal.evolving = false
      goal.evolving_started_at_ms = 0
      room_evolution[room].goals[goal_name] = goal
      dirty_rooms.incl(room)
      queue_goal_updated_outbox_locked(room, goal, now, outbox)
      sys_payload = system_msg_payload(username & " reset \"" & goal_name & "\" to gen 0")
      targets = room_targets(room)
      applied = true
  if applied:
    for item in outbox:
      broadcast(item.targets, item.payload)
    if targets.len > 0:
      broadcast(targets, sys_payload)

proc apply_goal_evolve*(room, goal_name, genomes_json: string; top_score: int) {.exportpy.} =
  var
    outbox: seq[PendingBroadcast] = @[]
    sys_payload = ""
    targets: seq[WebSocket]
    applied = false
  with_state_lock:
    if room in room_evolution and goal_name in room_evolution[room].goals:
      let now = now_ms()
      var goal = room_evolution[room].goals[goal_name]
      commit_next_generation_locked(room, goal, genomes_json, top_score, now, outbox, sys_payload)
      targets = room_targets(room)
      applied = true
  if applied:
    for item in outbox:
      broadcast(item.targets, item.payload)
    if targets.len > 0 and sys_payload.len > 0:
      broadcast(targets, sys_payload)

proc apply_goal_brood*(room, goal_name, brood_json: string; top_score: int) {.exportpy.} =
  var parsed: JsonNode
  try:
    parsed = parse_json(brood_json)
  except CatchableError:
    return
  var
    outbox: seq[PendingBroadcast] = @[]
    sys_payload = ""
    targets: seq[WebSocket] = @[]
    applied = false
  with_state_lock:
    if room in room_evolution and goal_name in room_evolution[room].goals:
      let now = now_ms()
      var goal = room_evolution[room].goals[goal_name]
      let bundle = parse_brood_bundle_node(parsed, max(1, goal.votes.len), goal.params.brood_size)
      if not bundle.ok:
        goal.evolving = false
        goal.evolving_started_at_ms = 0
        goal.sync_goal_countdown(cfg, now)
        room_evolution[room].goals[goal_name] = goal
        targets = room_targets(room)
        if targets.len > 0:
          outbox.add(PendingBroadcast(
            targets: targets,
            payload: build_votes_updated_payload(goal, cfg, clients, now, max(1, targets.len), is_public = (room == "lobby")),
          ))
        applied = true
      elif bundle.groups.len == 0:
        # No broods to vote on (e.g. all disliked -> fresh randoms): commit directly to next gen.
        var max_slot = max(0, bundle.pop_size - 1)
        for f in bundle.fixed:
          if f.slot > max_slot: max_slot = f.slot
        var slots = new_seq[string](max_slot + 1)
        for f in bundle.fixed:
          if f.slot >= 0 and f.slot < slots.len and f.genome_json.len > 0:
            slots[f.slot] = f.genome_json
        var filled: seq[string] = @[]
        for s in slots:
          if s.len > 0: filled.add(s)
        if filled.len > 0:
          commit_next_generation_locked(room, goal, "[" & filled.join(",") & "]", top_score, now, outbox, sys_payload)
          targets = room_targets(room)
          applied = true
      else:
        goal.brood_size_active = max(2, bundle.brood_size)
        goal.brood_pop_size = bundle.pop_size
        goal.brood_top_score = top_score
        goal.brood_fixed = bundle.fixed
        goal.brood_groups = bundle.groups
        goal.phase = "brood"
        goal.countdown_ends_at_ms = 0
        goal.evolving = false
        goal.evolving_started_at_ms = 0
        room_evolution[room].goals[goal_name] = goal
        dirty_rooms.incl(room)
        queue_goal_updated_outbox_locked(room, goal, now, outbox)
        sys_payload = system_msg_payload(
          "Brood selection open for \"" & goal_name & "\" (" &
          $goal.brood_groups.len & " broods × " & $goal.brood_size_active & " siblings)"
        )
        targets = room_targets(room)
        applied = true
  if applied:
    for item in outbox:
      broadcast(item.targets, item.payload)
    if targets.len > 0 and sys_payload.len > 0:
      broadcast(targets, sys_payload)

proc cancel_goal_evolve*(room, goal_name: string) {.exportpy, gcsafe.} =
  var
    votes_payload = ""
    targets: seq[WebSocket] = @[]
  with_state_lock:
    if room in room_evolution and goal_name in room_evolution[room].goals:
      var goal = room_evolution[room].goals[goal_name]
      if goal.evolving:
        let now = now_ms()
        goal.evolving = false
        goal.evolving_started_at_ms = 0
        goal.sync_goal_countdown(cfg, now)
        room_evolution[room].goals[goal_name] = goal
        targets = room_targets(room)
        if targets.len > 0:
          votes_payload = build_votes_updated_payload(
            goal, cfg, clients, now, max(1, targets.len), is_public = (room == "lobby")
          )
  if votes_payload.len > 0 and targets.len > 0:
    broadcast(targets, votes_payload)

proc cancel_goal_create*(room, goal_name: string) {.exportpy, gcsafe.} =
  with_state_lock:
    if room in room_evolution:
      room_evolution[room].pending_creates.excl(goal_name)
      room_evolution[room].pending_create_tasks.del(goal_name)
      room_evolution[room].pending_create_params.del(goal_name)
      room_evolution[room].pending_create_env_params.del(goal_name)

proc apply_goal_resize*(room, goal_name, username, genomes_json: string) {.exportpy.} =
  var
    outbox: seq[PendingBroadcast] = @[]
    sys_payload: string
    targets: seq[WebSocket]
    applied = false
  with_state_lock:
    if room in room_evolution and goal_name in room_evolution[room].goals:
      let now = now_ms()
      var goal = room_evolution[room].goals[goal_name]
      let tc = task_cfg_locked(goal.task)
      let indexed = index_goal_genomes(genomes_json, tc.sim_phase_enabled)
      let g_len = max(1, indexed.items.len)
      var next_votes = empty_votes(g_len)
      let keep = min(goal.votes.len, g_len)
      for i in 0 ..< keep:
        next_votes[i] = goal.votes[i]
      goal.genomes_json = genomes_json
      goal.sharded_genomes_json = indexed.sharded_json
      goal.genome_items = indexed.items
      goal.uuid_to_idx = indexed.uuid_map
      goal.head_uuid = indexed.head_uuid
      goal.lineage_json = merge_missing_lineage_json(goal.lineage_json, genomes_json, goal.generation)
      goal.refresh_goal_lineage_cache()
      goal.votes = next_votes
      goal.clear_goal_brood()
      if goal.phase == "brood":
        goal.phase = initial_phase_for_task(tc)
      goal.evolving = false
      goal.evolving_started_at_ms = 0
      goal.sync_goal_countdown(cfg, now)
      room_evolution[room].goals[goal_name] = goal
      dirty_rooms.incl(room)
      queue_goal_updated_outbox_locked(room, goal, now, outbox)
      sys_payload = system_msg_payload(username & " set \"" & goal_name & "\" population to " & $g_len)
      targets = room_targets(room)
      applied = true
  if applied:
    for item in outbox:
      broadcast(item.targets, item.payload)
    if targets.len > 0:
      broadcast(targets, sys_payload)

proc seed_room_history*(room, messages_json: string) {.exportpy.} =
  let parsed = parse_json(messages_json)
  if parsed.kind != JArray:
    return
  var items: seq[string] = @[]
  for item in parsed:
    items.add($item)
  if items.len > history_limit:
    items = items[items.len - history_limit .. ^1]
  with_state_lock:
    room_history[room] = items

proc drain_pending_messages*(): string {.exportpy.} =
  var batch: seq[PendingPersist]
  with_state_lock:
    if pending_saves.len == 0:
      return "[]"
    batch = move pending_saves
    pending_saves = @[]
  var arr = new_j_array()
  for item in batch:
    arr.add(%*{
      "room": item.room,
      "username": item.username,
      "message": item.message,
    })
  $arr

proc get_room_goal_pop_sizes*(room: string): string {.exportpy.} =
  ## Return JSON object mapping active goal names in `room` to their population sizes.
  var obj = new_j_object()
  with_state_lock:
    if room in room_evolution:
      for g_name in room_evolution[room].goal_order:
        if g_name in room_evolution[room].goals:
          obj[g_name] = %room_evolution[room].goals[g_name].votes.len
  $obj

proc has_room_evolution*(room: string): bool {.exportpy.} =
  with_state_lock:
    result = room in room_evolution and not room_evolution[room].initializing

proc drain_dirty_room_states*(): string {.exportpy.} =
  var arr = new_j_array()
  with_state_lock:
    if dirty_rooms.len == 0:
      return "[]"
    for room in dirty_rooms:
      if room in room_evolution and not room_evolution[room].initializing:
        arr.add(%*{
          "room": room,
          "state_json": serialize_room_evolution(room_evolution[room]),
        })
    dirty_rooms.clear()
  $arr

proc restore_room_state*(room, state_json: string) {.exportpy.} =
  if state_json.len == 0:
    return
  let det = json_detach_state_doc(state_json)
  var parsed: JsonNode
  try:
    parsed = parse_json(det.lean_doc)
  except CatchableError:
    return
  if parsed.kind != JObject:
    return
  let goals_node = parsed.get_or_default("goals")
  if goals_node == nil or goals_node.kind != JArray:
    return

  var outbox: seq[PendingBroadcast] = @[]
  with_state_lock:
    let dp = parse_goal_params_json(parsed.get_or_default("default_params"), cfg.params)
    var evo = RoomEvolution(
      goal_order: @[],
      goals: init_table[string, GoalPopulation](),
      pending_creates: init_hash_set[string](),
      pending_create_tasks: init_table[string, string](),
      pending_create_params: init_table[string, GoalParams](),
      pending_create_env_params: init_table[string, string](),
      default_params: dp,
      initializing: false,
    )
    for goal_idx in 0 ..< goals_node.len:
      let item = goals_node[goal_idx]
      if item.kind != JObject:
        continue
      let name = sanitize_goal_name(item.get_or_default("name").get_str(""))
      let task = sanitize_task_name_locked(item.get_or_default("task").get_str("image"))
      let g_json =
        if goal_idx < det.genomes_list.len: det.genomes_list[goal_idx]
        else: $item.get_or_default("genomes")
      if name.len == 0 or g_json.len <= 2:
        continue
      let tc = task_cfg_locked(task)
      let indexed = index_goal_genomes(g_json, tc.sim_phase_enabled)
      if indexed.items.len == 0:
        continue
      let pop_cnt = indexed.items.len
      let gen = max(0, item.get_or_default("generation").get_int(0))
      var gp = parse_goal_params_json(item.get_or_default("params"), dp)
      if gp.min_votes <= 0:
        gp.min_votes = default_min_votes_for_task_locked(task)
      clamp_control_min_votes_locked(task, gp)
      let env_node = item.get_or_default("env_params")
      let env_json =
        if env_node != nil and env_node.kind == JObject: $env_node
        else: "{}"
      let lin_node = item.get_or_default("lineage")
      let l_json =
        if lin_node != nil and lin_node.kind == JArray and lin_node.len > 0:
          $lin_node
        else:
          $extract_lineage_nodes_from_stubs(indexed.stubs_arr, gen)
      let lean = compute_lean_lineage(l_json, g_json, gen)
      let default_phase = initial_phase_for_task(tc)
      let raw_phase = item.get_or_default("phase").get_str(default_phase)
      let brood_bundle = parse_brood_bundle_node(
        item.get_or_default("brood_state"),
        max(1, pop_cnt),
        gp.brood_size,
      )
      let has_brood = raw_phase == "brood" and brood_bundle.ok and brood_bundle.groups.len > 0
      let eff_phase =
        if has_brood: "brood"
        elif raw_phase in ["simulating", "voting"]: raw_phase
        else: default_phase
      let eff_sim_done = clamp(item.get_or_default("sim_completed").get_int(0), 0, max(1, pop_cnt))
      let eff_auto = item.get_or_default("auto_evolve").get_bool(tc.default_auto_evolve)
      evo.goal_order.add(name)
      evo.goals[name] = GoalPopulation(
        name: name,
        task: task,
        generation: gen,
        genomes_json: g_json,
        sharded_genomes_json: indexed.sharded_json,
        genome_items: indexed.items,
        uuid_to_idx: indexed.uuid_map,
        head_uuid: indexed.head_uuid,
        env_params_json: env_json,
        lineage_json: l_json,
        lean_lineage_json: lean.lean_json,
        lineage_total: lean.total,
        votes: empty_votes(max(1, pop_cnt)),
        phase: eff_phase,
        sim_completed: eff_sim_done,
        auto_evolve: eff_auto,
        countdown_ends_at_ms: 0,
        evolving: false,
        evolving_started_at_ms: 0,
        params: gp,
        brood_size_active: (if has_brood: max(2, brood_bundle.brood_size) else: 0),
        brood_pop_size: (if has_brood: brood_bundle.pop_size else: 0),
        brood_top_score: (if has_brood: brood_bundle.top_score else: 0),
        brood_fixed: (if has_brood: brood_bundle.fixed else: @[]),
        brood_groups: (if has_brood: brood_bundle.groups else: @[]),
      )
    if evo.goal_order.len > 0 or room != "lobby":
      room_evolution[room] = evo
      dirty_rooms.excl(room)
      if room in rooms and rooms[room].len > 0:
        queue_evolution_state_outbox_locked(room, outbox)
  for item in outbox:
    broadcast(item.targets, item.payload)

proc set_admin_password*(pw: string) {.exportpy.} =
  with_state_lock:
    admin_password = pw

proc set_evolve_token*(token: string) {.exportpy.} =
  with_state_lock:
    evolve_token = token.strip()

proc clear_state*() {.exportpy.} =
  with_state_lock:
    cfg = default_runtime_config()
    task_configs = default_task_configs()
    admin_password = ""
    evolve_token = ""
    clients.clear()
    pending_upgrades.clear()
    next_upgrade_seq = 0
    rooms.clear()
    room_history.clear()
    room_evolution.clear()
    dirty_rooms.clear()
    pending_saves = @[]
    pending_evo_jobs = @[]
