## Mummy WebSocket server and room evolution state engine for direct-evolution-together.

import std/[algorithm, atomics, json, locks, os, sets, strutils, sysrand, tables, times]
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
  max_population_size = 64
  control_max_population_size = 32
  default_population_size = 30
  default_control_population_size = 30
  default_control_min_votes = 20
  max_round_min_votes = 256

  min_votes_per_user = 1
  max_votes_per_user = 64
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

  default_min_votes = 20
  default_countdown_ms = 8000'i64
  vote_burst_window_ms = 1000'i64
  vote_burst_max_ops = 64
  max_lineage_nodes = 300
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
    has_elite_override: bool
    default_elite_frac: float

  PendingClient = object
    room: string
    username: string
    cid: string
    is_reconnect: bool
    initial_goal: string
    initial_task: string
    initial_pop_size: int
    initial_params: GoalParams
    gzip: bool
    can_evolve_now: bool
    lazy_goals: bool
    active_goal: string

  ClientInfo = object
    ws: WebSocket
    room: string
    username: string
    cid: string
    last_seen_ms: int64
    last_ping_ms: int64
    vote_timestamps_ms: seq[int64]
    gzip: bool
    can_evolve_now: bool
    lazy_goals: bool
    active_goal: string

  PendingPersist = object
    room: string
    username: string
    message: string

  GoalPopulation = object
    name: string
    task: string
    generation: int
    genomes_json: string
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

  RoomEvolution = object
    goal_order: seq[string]
    goals: Table[string, GoalPopulation]
    pending_creates: HashSet[string]
    pending_create_tasks: Table[string, string]
    pending_create_params: Table[string, GoalParams]
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
  result = initTable[string, TaskConfig]()
  result["image"] = TaskConfig(
    is_control: false,
    default_pop_size: default_population_size,
    max_pop_size: max_population_size,
    default_min_votes: 0,
    has_elite_override: false,
    default_elite_frac: default_elite_frac,
  )
  result["cartpole"] = TaskConfig(
    is_control: true,
    default_pop_size: default_control_population_size,
    max_pop_size: control_max_population_size,
    default_min_votes: default_control_min_votes,
    has_elite_override: true,
    default_elite_frac: 0.0,
  )
  result["halfcheetah"] = TaskConfig(
    is_control: true,
    default_pop_size: default_control_population_size,
    max_pop_size: control_max_population_size,
    default_min_votes: default_control_min_votes,
    has_elite_override: true,
    default_elite_frac: 0.05,
  )
  let populora_cfg = TaskConfig(
    is_control: true,
    sim_phase_enabled: true,
    default_auto_evolve: false,
    default_pop_size: 8,
    max_pop_size: control_max_population_size,
    default_min_votes: 4,
    has_elite_override: true,
    default_elite_frac: 0.25,
  )
  for tid in ["microduck", "unitree_go2", "so101"]:
    result[tid] = populora_cfg
  result["external"] = TaskConfig(
    is_control: true,
    default_pop_size: default_control_population_size,
    max_pop_size: max_population_size,
    default_min_votes: default_control_min_votes,
    has_elite_override: true,
    default_elite_frac: 0.10,
  )

var
  state_lock: Lock
  ready_lock: Lock
  ready_cond: Cond

  cfg = default_runtime_config()
  task_configs = default_task_configs()
  clients: Table[WebSocket, ClientInfo]
  pending_upgrades: Table[WebSocket, PendingClient]
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

initLock(state_lock)
initLock(ready_lock)
initCond(ready_cond)

template with_state_lock(body: untyped) =
  withLock state_lock:
    {.gcsafe.}:
      body

proc quiet_logger(level: LogLevel; args: varargs[string]) {.gcsafe.} =
  if level == ErrorLevel:
    echoLogger(level, args)

proc now_ms(): int64 {.gcsafe, inline.} =
  let t = getTime()
  t.toUnix * 1000 + int64(t.nanosecond div 1_000_000)

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

proc sanitize_username(raw: string): string {.gcsafe.} =
  var cleaned = raw.strip()
  if cleaned.len > max_username_len:
    cleaned = cleaned[0 ..< max_username_len]
  if cleaned.len > 0:
    return cleaned
  let bytes = urandom(2)
  "anon-" & toHex(bytes[0]).toLowerAscii() & toHex(bytes[1]).toLowerAscii()

func sanitize_goal_name(raw: string): string =
  result = raw.strip()
  if result.len > max_goal_name_len:
    result = result[0 ..< max_goal_name_len]

func sanitize_cid(raw: string): string =
  let trimmed = raw.strip()
  if trimmed.len == 0:
    return ""
  result = newStringOfCap(min(trimmed.len, max_cid_len))
  for ch in trimmed:
    if ch in {'a'..'z', 'A'..'Z', '0'..'9', '_', '-'}:
      result.add(ch)
      if result.len >= max_cid_len:
        break

proc sanitize_task_name_locked(raw: string): string {.gcsafe.} =
  let cleaned = raw.strip().toLowerAscii()
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
    has_elite_override: false,
    default_elite_frac: default_elite_frac,
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

func clamp_goal_params(p: GoalParams): GoalParams {.inline.} =
  GoalParams(
    max_votes_per_user: clamp(p.max_votes_per_user, min_votes_per_user, max_votes_per_user),
    min_votes: (if p.min_votes > 0: clamp(p.min_votes, 1, max_round_min_votes) else: 0),
    mut_sigma: clamp(p.mut_sigma, min_mut_sigma, max_mut_sigma),
    elite_frac: clamp(p.elite_frac, min_elite_frac, max_elite_frac),
    crossover_rate: clamp(p.crossover_rate, min_crossover_rate, max_crossover_rate),
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
    clamp(parseInt(s), lo, hi)
  except CatchableError:
    fallback

proc parse_clamped_float(raw: string; fallback, lo, hi: float): float {.gcsafe.} =
  let s = raw.strip()
  if s.len == 0:
    return fallback
  try:
    clamp(parseFloat(s), lo, hi)
  except CatchableError:
    fallback

proc make_unique_username(
  room, desired: string;
  exclude_ws = default(WebSocket);
  has_exclude = false;
): string {.gcsafe.} =
  ## Ensure no two users in the same room share a username (case-insensitive).
  var taken = initHashSet[string]()
  {.gcsafe.}:
    if room in rooms:
      for ws in rooms[room]:
        if has_exclude and ws == exclude_ws:
          continue
        if ws in clients:
          taken.incl(clients[ws].username.toLowerAscii())

  if desired.toLowerAscii() notin taken:
    return desired

  var suffix = 2
  while true:
    let suffix_str = "-" & $suffix
    let max_base = max_username_len - suffix_str.len
    let base =
      if desired.len > max_base: desired[0 ..< max_base]
      else: desired
    let candidate = base & suffix_str
    if candidate.toLowerAscii() notin taken:
      return candidate
    inc suffix

func empty_votes(n: int): seq[Table[WebSocket, int]] =
  result = newSeq[Table[WebSocket, int]](n)
  for i in 0 ..< n:
    result[i] = initTable[WebSocket, int]()

func count_goal_votes(
  goal: GoalPopulation
): tuple[total_votes, voted_candidates, liked_candidates: int, stats: seq[VoteStats]] =
  let n = goal.votes.len
  result.total_votes = 0
  result.voted_candidates = 0
  result.liked_candidates = 0
  result.stats = newSeq[VoteStats](n)
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

func remaining_countdown_ms(goal: GoalPopulation; now: int64): int64 {.inline.} =
  if goal.countdown_ends_at_ms <= 0:
    return 0
  max(0'i64, goal.countdown_ends_at_ms - now)

proc sync_goal_countdown(goal: var GoalPopulation; c: RuntimeConfig; now: int64) {.gcsafe.} =
  if goal.evolving or goal.phase == "simulating":
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

proc extract_lineage_nodes(genomes_json: string; gen: int): JsonNode {.gcsafe.} =
  result = newJArray()
  if genomes_json.len == 0:
    return
  let parsed = parseJson(genomes_json)
  if parsed.kind != JArray:
    return
  for item in parsed:
    let gid = item.getOrDefault("id").getInt(0)
    let raw_parents = item.getOrDefault("parents")
    let parents_node =
      if raw_parents != nil and raw_parents.kind == JArray: raw_parents
      else: newJArray()
    let op_str = item.getOrDefault("op").getStr(if gen == 0: "random" else: "mutation")
    if item.getOrDefault("kind").getStr("") == "populora":
      result.add(%*{
        "id": gid,
        "gen": gen,
        "parents": parents_node,
        "op": op_str,
        "kind": "populora",
        "hash": item.getOrDefault("hash").getStr(""),
        "baseHash": item.getOrDefault("baseHash").getStr(""),
      })
    else:
      result.add(%*{
        "id": gid,
        "gen": gen,
        "parents": parents_node,
        "op": op_str,
        "weights": item.getOrDefault("weights"),
        "biases": item.getOrDefault("biases"),
        "activations": item.getOrDefault("activations"),
      })

proc compute_lean_lineage(lineage_json, genomes_json: string; gen: int): tuple[lean_json: string, total: int] {.gcsafe.} =
  ## Precompute active-ancestry JSON (omitting live generation weights) once per generation change.
  let parsed =
    if lineage_json.len > 0: parseJson(lineage_json)
    else: extract_lineage_nodes(genomes_json, gen)
  if parsed.kind != JArray:
    return ("[]", 0)
  result.total = parsed.len

  var by_id = initTable[int, JsonNode]()
  for node in parsed:
    by_id[node.getOrDefault("id").getInt(0)] = node

  var keep = initHashSet[int]()
  var queue: seq[int] = @[]
  for node in parsed:
    if node.getOrDefault("gen").getInt(-1) == gen:
      let gid = node.getOrDefault("id").getInt(0)
      if gid notin keep:
        keep.incl(gid)
        queue.add(gid)

  var q_idx = 0
  while q_idx < queue.len:
    let cur_id = queue[q_idx]
    inc q_idx
    if cur_id in by_id:
      for p in by_id[cur_id].getOrDefault("parents"):
        let pid = p.getInt(0)
        if pid notin keep and pid in by_id:
          keep.incl(pid)
          queue.add(pid)

  var arr = newJArray()
  for node in parsed:
    let gid = node.getOrDefault("id").getInt(0)
    if gid notin keep:
      continue
    if node.getOrDefault("gen").getInt(-1) == gen:
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
  result.lineage = newJArray()
  if not full and goal.lean_lineage_json.len > 0:
    result.lineage = parseJson(goal.lean_lineage_json)
    result.total = goal.lineage_total
    return
  let parsed =
    if goal.lineage_json.len > 0: parseJson(goal.lineage_json)
    else: extract_lineage_nodes(goal.genomes_json, goal.generation)
  if parsed.kind != JArray:
    return
  result.total = parsed.len
  if full:
    for node in parsed:
      result.lineage.add(node)
    return
  let fallback = compute_lean_lineage(goal.lineage_json, goal.genomes_json, goal.generation)
  result.lineage = parseJson(fallback.lean_json)
  result.total = fallback.total

proc merge_missing_lineage_json(prev_lineage_json, genomes_json: string; gen: int): string {.gcsafe.} =
  ## Append any newly added genomes (e.g. from population resize) without duplicating existing IDs.
  let fresh = extract_lineage_nodes(genomes_json, gen)
  var combined =
    if prev_lineage_json.len > 0: parseJson(prev_lineage_json)
    else: newJArray()
  if combined.kind != JArray:
    combined = newJArray()
  var existing_ids = initHashSet[int]()
  for node in combined:
    existing_ids.incl(node.getOrDefault("id").getInt(0))
  for node in fresh:
    let gid = node.getOrDefault("id").getInt(0)
    if gid notin existing_ids:
      existing_ids.incl(gid)
      combined.add(node)
  $combined

proc merge_lineage_json(prev_lineage_json, genomes_json: string; gen: int): string {.gcsafe.} =
  let fresh = extract_lineage_nodes(genomes_json, gen)
  var combined =
    if prev_lineage_json.len > 0: parseJson(prev_lineage_json)
    else: newJArray()
  if combined.kind != JArray:
    combined = newJArray()
  for node in fresh:
    combined.add(node)

  if combined.len > max_lineage_nodes:
    ## Keep ancestor chain reachable from the latest generation, bounded to max_lineage_nodes.
    var by_id = initTable[int, JsonNode]()
    for node in combined:
      by_id[node["id"].getInt()] = node
    var keep_ids = initHashSet[int]()
    var queue: seq[int] = @[]
    for node in fresh:
      let gid = node["id"].getInt()
      if gid notin keep_ids:
        keep_ids.incl(gid)
        queue.add(gid)

    var q_idx = 0
    while q_idx < queue.len:
      let cur_id = queue[q_idx]
      inc q_idx
      if cur_id in by_id:
        for p in by_id[cur_id].getOrDefault("parents"):
          let pid = p.getInt()
          if pid notin keep_ids and pid in by_id:
            keep_ids.incl(pid)
            queue.add(pid)

    var pruned = newJArray()
    for node in combined:
      if node["id"].getInt() in keep_ids:
        pruned.add(node)
    if pruned.len > max_lineage_nodes:
      var trimmed = newJArray()
      let start_idx = pruned.len - max_lineage_nodes
      for i in start_idx ..< pruned.len:
        trimmed.add(pruned[i])
      pruned = trimmed
    combined = pruned
  $combined

proc build_votes_json(
  goal: GoalPopulation;
  client_map: Table[WebSocket, ClientInfo];
): tuple[votes_arr: JsonNode, total_votes, voted_candidates, liked_candidates: int] {.gcsafe.} =
  let counted = count_goal_votes(goal)
  result.total_votes = counted.total_votes
  result.voted_candidates = counted.voted_candidates
  result.liked_candidates = counted.liked_candidates
  result.votes_arr = newJArray()
  for i in 0 ..< goal.votes.len:
    var up_users: seq[string] = @[]
    var down_users: seq[string] = @[]
    for ws, v in goal.votes[i]:
      if ws in client_map:
        if v > 0:
          up_users.add(client_map[ws].username)
        elif v < 0:
          down_users.add(client_map[ws].username)
    up_users.sort(cmpIgnoreCase)
    down_users.sort(cmpIgnoreCase)
    let s = counted.stats[i]
    result.votes_arr.add(%*{
      "up": s.up,
      "down": s.down,
      "score": s.score,
      "upUsers": up_users,
      "downUsers": down_users,
    })

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
    obj["totalVotes"] = %vinfo.total_votes
    obj["votedCandidates"] = %vinfo.voted_candidates
    obj["likedCandidates"] = %vinfo.liked_candidates
  else:
    let counted = count_goal_votes(goal)
    obj["votes"] = newJArray()
    obj["totalVotes"] = %counted.total_votes
    obj["votedCandidates"] = %counted.voted_candidates
    obj["likedCandidates"] = %counted.liked_candidates
  obj["minVotes"] = %effective_min_votes(goal, c)
  obj["maxVotesPerUser"] = %effective_max_votes(goal, c, is_pub_image)
  let tc = task_cfg_locked(goal.task)
  let eff_phase =
    if goal.evolving: "evolving"
    elif goal.phase.len > 0: goal.phase
    elif tc.sim_phase_enabled: "simulating"
    else: "voting"
  obj["phase"] = %eff_phase
  obj["simPhaseEnabled"] = %tc.sim_phase_enabled
  obj["simCompleted"] = %goal.sim_completed
  obj["simTotal"] = %max(1, goal.votes.len)
  obj["autoEvolve"] = %goal.auto_evolve
  obj["countdownMs"] = %remaining_countdown_ms(goal, now)
  obj["countdownDurationMs"] = %c.countdown_ms
  obj["evolving"] = %goal.evolving
  obj["mutationSigma"] = %goal.params.mut_sigma
  obj["eliteFrac"] = %goal.params.elite_frac
  obj["crossoverRate"] = %goal.params.crossover_rate
  obj["populationSize"] = %goal.votes.len
  obj["maxPopulationSize"] = %tc.max_pop_size

proc goal_to_json_str(
  goal: GoalPopulation;
  c: RuntimeConfig;
  client_map: Table[WebSocket, ClientInfo];
  now: int64;
  room_user_count: int = 1;
  is_public: bool = false;
  include_genomes: bool = true;
): string {.gcsafe.} =
  ## Fast zero-reparse serializer: splices pre-serialized `genomes_json` and `lean_lineage_json` directly.
  let round_obj = newJObject()
  round_obj.attach_round_fields(
    goal, c, client_map, now, room_user_count, is_public, include_votes = include_genomes
  )
  let round_str = $round_obj
  let genomes_str =
    if include_genomes and goal.genomes_json.len > 0: goal.genomes_json
    else: "[]"
  let lineage_str =
    if include_genomes and goal.lean_lineage_json.len > 0: goal.lean_lineage_json
    else: "[]"
  let task_str = if goal.task.len > 0: goal.task else: "image"

  result = newStringOfCap(genomes_str.len + lineage_str.len + round_str.len + 192)
  result.add("{\"name\":")
  result.add(escapeJson(goal.name))
  result.add(",\"task\":")
  result.add(escapeJson(task_str))
  result.add(",\"generation\":")
  result.add($goal.generation)
  result.add(",\"genomesLoaded\":")
  result.add(if include_genomes: "true" else: "false")
  result.add(",\"genomes\":")
  result.add(genomes_str)
  result.add(",\"lineage\":")
  result.add(lineage_str)
  result.add(",\"lineageTotal\":")
  result.add($goal.lineage_total)
  result.add(",")
  result.add(round_str[1 .. ^1])

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
): string {.gcsafe.} =
  let g_str = goal_to_json_str(
    goal, c, client_map, now, room_user_count,
    is_public = is_public, include_genomes = include_genomes,
  )
  result = newStringOfCap(g_str.len + 32)
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
            is_public = is_pub, include_genomes = inc_genomes,
          )
          total_len += g_str.len + 1
          parts.add(g_str)
    result = newStringOfCap(total_len)
    result.add("{\"type\":\"evolution_state\",\"goals\":[")
    for idx, p in parts:
      if idx > 0:
        result.add(",")
      result.add(p)
    result.add("]}")

proc goal_params_json(p: GoalParams): JsonNode {.gcsafe.} =
  %*{
    "maxVotesPerUser": p.max_votes_per_user,
    "minVotes": p.min_votes,
    "mutSigma": p.mut_sigma,
    "eliteFrac": p.elite_frac,
    "crossoverRate": p.crossover_rate,
  }

proc parse_goal_params_json(node: JsonNode; fallback: GoalParams): GoalParams {.gcsafe.} =
  if node == nil or node.kind != JObject:
    return fallback
  clamp_goal_params(GoalParams(
    max_votes_per_user: node.getOrDefault("maxVotesPerUser").getInt(fallback.max_votes_per_user),
    min_votes: node.getOrDefault("minVotes").getInt(fallback.min_votes),
    mut_sigma: node.getOrDefault("mutSigma").getFloat(fallback.mut_sigma),
    elite_frac: node.getOrDefault("eliteFrac").getFloat(fallback.elite_frac),
    crossover_rate: node.getOrDefault("crossoverRate").getFloat(fallback.crossover_rate),
  ))

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
    let task_str = if g.task.len > 0: g.task else: "image"
    var item = newStringOfCap(g_genomes.len + g_lineage.len + gp_str.len + 128)
    item.add("{\"name\":")
    item.add(escapeJson(g.name))
    item.add(",\"task\":")
    item.add(escapeJson(task_str))
    item.add(",\"generation\":")
    item.add($g.generation)
    item.add(",\"genomes\":")
    item.add(g_genomes)
    item.add(",\"lineage\":")
    item.add(g_lineage)
    item.add(",\"params\":")
    item.add(gp_str)
    item.add(",\"phase\":")
    item.add(escapeJson(if g.phase.len > 0: g.phase else: "voting"))
    item.add(",\"simCompleted\":")
    item.add($g.sim_completed)
    item.add(",\"autoEvolve\":")
    item.add(if g.auto_evolve: "true" else: "false")
    item.add("}")
    total_len += item.len + 1
    parts.add(item)
  result = newStringOfCap(total_len)
  result.add("{\"defaultParams\":")
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
  users.sort(cmpIgnoreCase)
  $(%*{
    "type": "presence",
    "users": users,
    "count": users.len,
  })

proc build_history_payload(room: string): string {.gcsafe.} =
  var parts: seq[string] = @[]
  {.gcsafe.}:
    if room in room_history:
      parts = room_history[room]
  "{\"type\":\"history\",\"messages\":[" & parts.join(",") & "]}"

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
    var full_targets: seq[WebSocket] = @[]
    var lazy_buckets = initTable[string, seq[WebSocket]]()
    let has_evo = room in room_evolution
    for ws in targets:
      if ws in clients and clients[ws].lazy_goals:
        let resolved =
          if has_evo: resolve_lazy_target_goal(room_evolution[room], clients[ws].active_goal)
          else: ""
        clients[ws].active_goal = resolved
        if resolved notin lazy_buckets:
          lazy_buckets[resolved] = @[]
        lazy_buckets[resolved].add(ws)
      else:
        full_targets.add(ws)
    if full_targets.len > 0:
      outbox.add(PendingBroadcast(
        targets: full_targets,
        payload: build_evolution_state_payload(room, lazy_goals = false),
      ))
    for focus_name, bucket_targets in lazy_buckets:
      outbox.add(PendingBroadcast(
        targets: bucket_targets,
        payload: build_evolution_state_payload(room, lazy_goals = true, focus_goal = focus_name),
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
    var full_targets: seq[WebSocket] = @[]
    var stub_targets: seq[WebSocket] = @[]
    for ws in targets:
      if ws in clients and clients[ws].lazy_goals and clients[ws].active_goal != goal.name:
        stub_targets.add(ws)
      else:
        full_targets.add(ws)
    if full_targets.len > 0:
      outbox.add(PendingBroadcast(
        targets: full_targets,
        payload: build_goal_updated_payload(
          goal, cfg, clients, now, r_cnt, is_public = is_pub, include_genomes = true
        ),
      ))
    if stub_targets.len > 0:
      outbox.add(PendingBroadcast(
        targets: stub_targets,
        payload: build_goal_updated_payload(
          goal, cfg, clients, now, r_cnt, is_public = is_pub, include_genomes = false
        ),
      ))

proc client_uses_gzip(ws: WebSocket): bool {.gcsafe.} =
  with_state_lock:
    result = ws in clients and clients[ws].gzip

proc compress_payload(payload: string): string {.gcsafe.} =
  try:
    result = compress(payload, DefaultCompression, dfGzip)
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
): GoalPopulation {.gcsafe.} =
  let clean_task = sanitize_task_name_locked(task)
  let tc = task_cfg_locked(clean_task)
  var params = gp
  clamp_control_min_votes_locked(clean_task, params)
  let l_json = merge_lineage_json("", genomes_json, 0)
  let lean = compute_lean_lineage(l_json, genomes_json, 0)
  let init_phase = if tc.sim_phase_enabled: "simulating" else: "voting"
  let init_auto =
    if auto_evolve_override >= 0: auto_evolve_override > 0
    else: tc.default_auto_evolve
  GoalPopulation(
    name: name,
    task: clean_task,
    generation: 0,
    genomes_json: genomes_json,
    lineage_json: l_json,
    lean_lineage_json: lean.lean_json,
    lineage_total: lean.total,
    votes: empty_votes(max(1, pop_len)),
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
): RoomEvolution {.gcsafe.} =
  result = RoomEvolution(
    goal_order: @[],
    goals: initTable[string, GoalPopulation](),
    pending_creates: initHashSet[string](),
    pending_create_tasks: initTable[string, string](),
    pending_create_params: initTable[string, GoalParams](),
    default_params: dp,
    initializing: false,
  )
  for item in parsed:
    let name = item["name"].getStr()
    let task = sanitize_task_name_locked(item.getOrDefault("task").getStr("image"))
    let g_arr = item["genomes"]
    let g_json = $g_arr
    var gp = dp
    let item_mv = item.getOrDefault("minVotes").getInt(-1)
    if item_mv > 0:
      gp.min_votes = clamp(item_mv, 1, max_round_min_votes)
    elif gp.min_votes <= 0:
      gp.min_votes = default_min_votes_for_task_locked(task)
    if item.hasKey("eliteFrac") and item["eliteFrac"].kind in {JFloat, JInt}:
      gp.elite_frac = clamp(item["eliteFrac"].getFloat(gp.elite_frac), min_elite_frac, max_elite_frac)
    elif task != "image":
      gp.elite_frac = default_elite_frac_for_task_locked(task, dp.elite_frac)
    let auto_override =
      if item.hasKey("autoEvolve") and item["autoEvolve"].kind == JBool:
        (if item["autoEvolve"].getBool(false): 1 else: 0)
      else: -1
    result.goal_order.add(name)
    result.goals[name] = init_goal_population_locked(name, task, g_json, g_arr.len, gp, auto_override)

proc upgrade_handler(request: Request) {.gcsafe.} =
  let room = request.pathParams["room"]
  if not is_valid_room(room):
    request.respond(400, body = "Invalid room name")
    return

  let username = sanitize_username(request.queryParams["username"])
  let cid = sanitize_cid(request.queryParams["cid"])
  let is_reconnect = request.queryParams["reconnect"] == "1"
  let initial_goal = sanitize_goal_name(request.queryParams["goal"])
  let raw_focus_goal = sanitize_goal_name(request.queryParams["focus"])
  let focus_goal = if raw_focus_goal.len > 0: raw_focus_goal else: initial_goal
  let lazy_goals = request.queryParams["lazy"] == "1"
  let raw_initial_task = request.queryParams["task"]
  let wants_gzip = request.queryParams["gzip"] == "1"
  let raw_evolve_token = request.queryParams["evolve_token"].strip()
  var
    base_cfg: RuntimeConfig
    server_evolve_token: string
    initial_task = "image"
    task_max_pop = max_population_size
    task_default_pop = default_population_size
    task_default_min_votes = 0
    task_default_elite = default_elite_frac
  with_state_lock:
    base_cfg = cfg
    server_evolve_token = evolve_token
    initial_task = sanitize_task_name_locked(raw_initial_task)
    task_max_pop = max_pop_for_task_locked(initial_task)
    task_default_pop = default_pop_for_task_locked(initial_task, base_cfg.population_size)
    task_default_min_votes = default_min_votes_for_task_locked(initial_task)
    task_default_elite = default_elite_frac_for_task_locked(initial_task, base_cfg.params.elite_frac)
  let can_evolve =
    server_evolve_token.len > 0 and
    constant_time_eq(raw_evolve_token, server_evolve_token)

  let req_pop = parse_clamped_int(
    request.queryParams["pop"],
    task_default_pop, min_population_size, task_max_pop,
  )
  let raw_min_votes = request.queryParams["min_votes"]
  let req_min_votes =
    if raw_min_votes.strip().len > 0:
      parse_clamped_int(raw_min_votes, task_default_min_votes, 1, max_round_min_votes)
    else:
      task_default_min_votes
  let req_params = GoalParams(
    max_votes_per_user: parse_clamped_int(
      request.queryParams["max_votes"],
      base_cfg.params.max_votes_per_user, min_votes_per_user, max_votes_per_user,
    ),
    min_votes: req_min_votes,
    mut_sigma: parse_clamped_float(
      request.queryParams["sigma"],
      base_cfg.params.mut_sigma, min_mut_sigma, max_mut_sigma,
    ),
    elite_frac: parse_clamped_float(
      request.queryParams["elite_frac"],
      task_default_elite, min_elite_frac, max_elite_frac,
    ),
    crossover_rate: parse_clamped_float(
      request.queryParams["crossover"],
      base_cfg.params.crossover_rate, min_crossover_rate, max_crossover_rate,
    ),
  )

  with_state_lock:
    let ws = request.upgradeToWebSocket()
    pending_upgrades[ws] = PendingClient(
      room: room,
      username: username,
      cid: cid,
      is_reconnect: is_reconnect,
      initial_goal: initial_goal,
      initial_task: initial_task,
      initial_pop_size: req_pop,
      initial_params: req_params,
      gzip: wants_gzip,
      can_evolve_now: can_evolve,
      lazy_goals: lazy_goals,
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
        for i in 0 ..< goal.votes.len:
          if ws in goal.votes[i]:
            goal.votes[i].del(ws)
            changed = true
        if changed:
          goal.sync_goal_countdown(cfg, now)
          room_evolution[info.room].goals[g_name] = goal
          vote_payloads.add(
            build_votes_updated_payload(goal, cfg, clients, now, r_cnt, is_public = (info.room == "lobby"))
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
    has_evolution_ready = false

  with_state_lock:
    if ws in pending_upgrades:
      let info = pending_upgrades[ws]
      pending_upgrades.del(ws)
      let now = now_ms()
      var replaced_same_room = false
      var prev_same_room_name = ""
      var same_room_old_sockets: seq[WebSocket] = @[]

      if info.cid.len >= 8:
        var matching_old: seq[WebSocket] = @[]
        for existing_ws, existing_info in clients:
          if existing_ws != ws and existing_info.cid == info.cid:
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
        last_seen_ms: now,
        last_ping_ms: now,
        gzip: info.gzip,
        can_evolve_now: info.can_evolve_now,
        lazy_goals: info.lazy_goals,
        active_goal: resolved_active_goal,
      )
      if info.room notin rooms:
        rooms[info.room] = initHashSet[WebSocket]()
      rooms[info.room].incl(ws)

      if same_room_old_sockets.len > 0 and info.room in room_evolution:
        let r_cnt = max(1, rooms[info.room].len)
        for g_name in room_evolution[info.room].goal_order:
          var goal = room_evolution[info.room].goals[g_name]
          var changed = false
          var transferred = false
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
          if changed:
            goal.sync_goal_countdown(cfg, now)
            room_evolution[info.room].goals[g_name] = goal
            extra_outbox.add(PendingBroadcast(
              targets: room_targets(info.room, ws, has_exclude = true),
              payload: build_votes_updated_payload(goal, cfg, clients, now, r_cnt, is_public = (info.room == "lobby")),
            ))
          elif transferred:
            room_evolution[info.room].goals[g_name] = goal
            if prev_same_room_name != unique_name:
              extra_outbox.add(PendingBroadcast(
                targets: room_targets(info.room, ws, has_exclude = true),
                payload: build_votes_updated_payload(goal, cfg, clients, now, r_cnt, is_public = (info.room == "lobby")),
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
        "canEvolveNow": info.can_evolve_now,
      })
      history_payload = build_history_payload(info.room)

      if info.room in room_evolution and not room_evolution[info.room].initializing:
        evolution_payload = build_evolution_state_payload(
          info.room,
          lazy_goals = info.lazy_goals,
          focus_goal = resolved_active_goal,
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
          goals: initTable[string, GoalPopulation](),
          pending_creates: initHashSet[string](),
          pending_create_tasks: initTable[string, string](),
          pending_create_params: initTable[string, GoalParams](),
          default_params: room_params,
          initializing: true,
        )
        pending_evo_jobs.add(EvoJob(
          kind: EvoJobKind.init_room,
          room: info.room,
          goal: (if is_private: info.initial_goal else: ""),
          task: (if is_private: info.initial_task else: "image"),
          username: unique_name,
          population_size: room_pop,
          params: room_params,
        ))

      presence_payload = build_presence_payload(info.room)
      if not (replaced_same_room and prev_same_room_name == unique_name):
        join_payload = system_msg_payload(unique_name & " joined")
      targets = room_targets(info.room)
      found = true

  for sws in stale_sockets:
    sws.close()
  for item in extra_outbox:
    broadcast(item.targets, item.payload)
  if found:
    send_payload(ws, identity_payload)
    send_payload(ws, history_payload)
    if has_evolution_ready:
      send_payload(ws, evolution_payload)
    broadcast(targets, presence_payload)
    if join_payload.len > 0:
      broadcast(targets, join_payload)

proc handle_rename(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  let desired_name = sanitize_username(payload.getOrDefault("username").getStr(""))
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
    changed = false

  with_state_lock:
    if ws in clients:
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
        targets = room_targets(room)
        other_targets = room_targets(room, ws, has_exclude = true)
        changed = true

  if changed:
    send_payload(ws, $(%*{
      "type": "identity",
      "username": assigned_name,
      "canEvolveNow": can_evolve,
    }))
    broadcast(other_targets, cursor_leave_payload)
    broadcast(targets, presence_payload)
    broadcast(targets, rename_payload)
    for vp in vote_payloads:
      broadcast(targets, vp)

proc handle_cursor(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  let x = clamp(payload.getOrDefault("x").getFloat(0.0), -0.05, 1.05)
  let y = clamp(payload.getOrDefault("y").getFloat(0.0), -0.05, 1.05)
  let visible = payload.getOrDefault("visible").getBool(true)
  let goal_name = sanitize_goal_name(payload.getOrDefault("goal").getStr(""))
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
  var text = payload.getOrDefault("message").getStr("").strip()
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
  let goal_name = sanitize_goal_name(payload.getOrDefault("goal").getStr(""))
  if goal_name.len == 0:
    return
  let raw_task = payload.getOrDefault("task").getStr("image")

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
          payload.getOrDefault("population_size").getInt(default_pop),
          min_population_size, max_pop_for_task_locked(task_name),
        )
        let req_min_votes = payload.getOrDefault("min_votes").getInt(default_min_votes_for_task_locked(task_name))
        let task_default_elite = default_elite_frac_for_task_locked(task_name, d.elite_frac)
        let g_params = clamp_goal_params(GoalParams(
          max_votes_per_user: payload.getOrDefault("max_votes_per_user").getInt(d.max_votes_per_user),
          min_votes: req_min_votes,
          mut_sigma: payload.getOrDefault("mutation_sigma").getFloat(d.mut_sigma),
          elite_frac: payload.getOrDefault("elite_frac").getFloat(task_default_elite),
          crossover_rate: payload.getOrDefault("crossover_rate").getFloat(d.crossover_rate),
        ))
        room_evolution[info.room].pending_creates.incl(goal_name)
        room_evolution[info.room].pending_create_tasks[goal_name] = task_name
        room_evolution[info.room].pending_create_params[goal_name] = g_params
        clients[ws].active_goal = goal_name
        pending_evo_jobs.add(EvoJob(
          kind: EvoJobKind.create,
          room: info.room,
          goal: goal_name,
          task: task_name,
          username: info.username,
          population_size: req_pop,
          params: g_params,
        ))

proc handle_cast_vote(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  let goal_name = payload.getOrDefault("goal").getStr("")
  let idx = payload.getOrDefault("index").getInt(-1)
  let req_vote = payload.getOrDefault("vote").getInt(0)
  let has_req_gen = payload.hasKey("generation")
  let req_gen = if has_req_gen: payload.getOrDefault("generation").getInt(-1) else: -1
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
            "maxVotesPerUser": effective_max_votes(goal, cfg, is_public_image_room(client_info.room, goal)),
            "message": "Simulations still running — voting unlocks once all candidates are evaluated",
          })
        elif not goal.evolving and idx < goal.votes.len and (not has_req_gen or req_gen == goal.generation):
          let now = now_ms()
          let prev_vote = goal.votes[idx].getOrDefault(ws, 0)
          let next_vote = if prev_vote == req_vote: 0 else: req_vote
          let cap = effective_max_votes(
            goal, cfg, is_public_image_room(client_info.room, goal)
          )
          if prev_vote == 0 and next_vote != 0 and count_user_active_votes(goal, ws) >= cap:
            throttle_payload = $(%*{
              "type": "vote_throttled",
              "goal": goal_name,
              "index": idx,
              "maxVotesPerUser": cap,
              "message": "Vote limit reached (" & $cap & " active votes per round — unvote one to switch)",
            })
          elif not allow_vote_burst(client_info, now):
            clients[ws] = client_info
            throttle_payload = $(%*{
              "type": "vote_throttled",
              "goal": goal_name,
              "index": idx,
              "maxVotesPerUser": cap,
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
  let goal_name = sanitize_goal_name(payload.getOrDefault("goal").getStr(""))
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
          "lineageTotal": lineage_payload.total,
        })
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
  var scores = newSeq[int](pop_len)
  var up_counts = newSeq[int](pop_len)
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
  let goal_name = sanitize_goal_name(payload.getOrDefault("goal").getStr(""))
  if goal_name.len == 0:
    return
  let enabled = payload.getOrDefault("enabled").getBool(false)
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

proc handle_evolve_now(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  let goal_name = payload.getOrDefault("goal").getStr("")
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
  let goal_name = sanitize_goal_name(payload.getOrDefault("goal").getStr(""))
  if goal_name.len == 0:
    return
  let need_genomes = payload.getOrDefault("needGenomes").getBool(true)
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
      payload = parseJson(message.data)
    except CatchableError:
      return
    if payload.kind != JObject:
      return
    case payload.getOrDefault("type").getStr("chat")
    of "ping": send_payload(ws, """{"type":"pong"}""")
    of "leave": handle_leave(ws)
    of "rename": handle_rename(ws, payload)
    of "cursor": handle_cursor(ws, payload)
    of "chat": handle_chat(ws, payload)
    of "create_goal": handle_create_goal(ws, payload)
    of "select_goal": handle_select_goal(ws, payload)
    of "cast_vote": handle_cast_vote(ws, payload)
    of "set_auto_evolve": handle_set_auto_evolve(ws, payload)
    of "evolve_now": handle_evolve_now(ws, payload)
    of "lineage_all": handle_lineage_all(ws, payload)
    else: discard
  of ErrorEvent:
    discard
  of CloseEvent:
    on_close(ws)

proc ticker_thread_proc(unused: int) {.thread.} =
  {.gcsafe.}:
    while not stop_ticker.load(moAcquire):
      sleep(ticker_sleep_ms)
      if stop_ticker.load(moAcquire):
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
  let room = request.pathParams["room"]
  if not is_valid_room(room):
    request.respond(400, body = "Invalid room name")
    return
  var obj = newJObject()
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
    parsed = parseJson(request.body)
  except CatchableError:
    request.respond(400, body = "Invalid JSON")
    return
  let room = parsed.getOrDefault("room").getStr("")
  let goal_name = parsed.getOrDefault("goal").getStr("")
  let username = parsed.getOrDefault("username").getStr("admin")
  let genomes_node = parsed.getOrDefault("genomes")
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
    parsed = parseJson(request.body)
  except CatchableError:
    request.respond(400, body = "Invalid JSON")
    return
  let room = parsed.getOrDefault("room").getStr("")
  let goal_name = parsed.getOrDefault("goal").getStr("")
  let username = parsed.getOrDefault("username").getStr("admin")
  let goals_node = parsed.getOrDefault("goals")
  if room.len == 0 or goals_node == nil or goals_node.kind != JArray:
    request.respond(400, body = "Missing room or goals")
    return
  clear_room_evolution(room, $goals_node, goal_name, username)
  request.respond(200, @[("Content-Type", "application/json")], """{"ok":true}""")

proc build_room_snapshot_json(room: string; filter_goal: string = ""): string {.gcsafe.} =
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
          let g_genomes = if g.genomes_json.len > 0: g.genomes_json else: "[]"
          let task_str = if g.task.len > 0: g.task else: "image"
          var item = newStringOfCap(g_genomes.len + 96)
          item.add("{\"name\":")
          item.add(escapeJson(g.name))
          item.add(",\"task\":")
          item.add(escapeJson(task_str))
          item.add(",\"generation\":")
          item.add($g.generation)
          item.add(",\"phase\":")
          item.add(escapeJson(if g.evolving: "evolving" elif g.phase.len > 0: g.phase else: "voting"))
          item.add(",\"simCompleted\":")
          item.add($g.sim_completed)
          item.add(",\"autoEvolve\":")
          item.add(if g.auto_evolve: "true" else: "false")
          item.add(",\"genomes\":")
          item.add(g_genomes)
          item.add("}")
          total_len += item.len + 1
          parts.add(item)
  result = newStringOfCap(total_len)
  result.add("{\"room\":")
  result.add(escapeJson(room))
  result.add(",\"goals\":[")
  for idx, p in parts:
    if idx > 0:
      result.add(",")
    result.add(p)
  result.add("]}")

proc get_room_snapshot_json*(room: string): string {.exportpy, gcsafe.} =
  ## Return current active goal states (name, task, generation, genomes) for `room`.
  build_room_snapshot_json(room)

proc get_room_goal_snapshot_json*(room, goal_name: string): string {.exportpy, gcsafe.} =
  ## Return snapshot for a single goal in `room` without serializing unrelated tabs.
  build_room_snapshot_json(room, sanitize_goal_name(goal_name))

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
  let room = request.pathParams["room"]
  if not is_valid_room(room):
    request.respond(400, body = "Invalid room name")
    return
  let goal_filter = sanitize_goal_name(request.queryParams["goal"])
  request.respond(200, @[("Content-Type", "application/json")], build_room_snapshot_json(room, goal_filter))

proc internal_room_broadcast_handler(request: Request) {.gcsafe.} =
  let auth = verify_internal_admin(request)
  if not auth.ok:
    request.respond(auth.code, body = auth.msg)
    return
  var parsed: JsonNode
  try:
    parsed = parseJson(request.body)
  except CatchableError:
    request.respond(400, body = "Invalid JSON")
    return
  let room = parsed.getOrDefault("room").getStr("")
  let goal_filter = sanitize_goal_name(parsed.getOrDefault("goal").getStr(""))
  let payload_node = parsed.getOrDefault("payload")
  if room.len == 0 or payload_node == nil or payload_node.kind != JObject:
    request.respond(400, body = "Missing room or payload")
    return
  broadcast_goal_payload(room, goal_filter, $payload_node)
  request.respond(200, @[("Content-Type", "application/json")], """{"ok":true}""")

proc get_room_peer_count*(room: string): int {.exportpy, gcsafe.} =
  ## Return number of unique active client sessions currently connected to `room`.
  with_state_lock:
    if room in rooms:
      var uniq_cids = initHashSet[string]()
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
          dirty_rooms.incl(room)
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
  let room = request.pathParams["room"]
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
    parsed = parseJson(request.body)
  except CatchableError:
    request.respond(400, body = "Invalid JSON")
    return
  let room = parsed.getOrDefault("room").getStr("")
  let goal_name = parsed.getOrDefault("goal").getStr("")
  let generation = parsed.getOrDefault("generation").getInt(0)
  let verified_count = parsed.getOrDefault("verifiedCount").getInt(0)
  let phase = notify_goal_sim_progress(room, goal_name, generation, verified_count)
  request.respond(200, @[("Content-Type", "application/json")], $(%*{"ok": true, "phase": phase}))

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
      router.get("/internal/room-peers/@room", internal_room_peers_handler)
      router.post("/internal/room-broadcast", internal_room_broadcast_handler)
      router.post("/internal/sim-progress", internal_sim_progress_handler)
      router.post("/internal/goal-reset", internal_goal_reset_handler)
      router.post("/internal/room-clear", internal_room_clear_handler)
      srv = newServer(router, websocket_handler, logHandler = quiet_logger)
      withLock ready_lock:
        server_ptr = srv
        server_ready.store(true, moRelease)
      signal(ready_cond)
      srv.serve(Port(args.port), args.host)
    except CatchableError:
      withLock ready_lock:
        server_failed.store(true, moRelease)
        server_ready.store(true, moRelease)
      signal(ready_cond)

# --- Python exports ---

proc start_server*(host: string; port: int) {.exportpy.} =
  var srv: Server = nil
  with_state_lock:
    if is_running:
      return
    server_ready.store(false, moRelease)
    server_failed.store(false, moRelease)
    stop_ticker.store(false, moRelease)
    server_ptr = nil
    is_running = true
    createThread(server_thread, serve_thread_proc, (host, port))

  withLock ready_lock:
    while not server_ready.load(moAcquire):
      wait(ready_cond, ready_lock)
    srv = server_ptr

  if server_failed.load(moAcquire) or srv == nil:
    with_state_lock:
      is_running = false
    joinThread(server_thread)
    raise newException(CatchableError, "Failed to start Mummy WebSocket server")

  srv.waitUntilReady()
  createThread(ticker_thread, ticker_thread_proc, 0)

proc stop_server*() {.exportpy.} =
  var srv: Server = nil
  var should_join = false
  with_state_lock:
    if is_running and server_ptr != nil:
      srv = server_ptr
      is_running = false
      should_join = true
  if should_join:
    stop_ticker.store(true, moRelease)
    joinThread(ticker_thread)
  if srv != nil:
    srv.close()
  if should_join:
    joinThread(server_thread)
  with_state_lock:
    server_ptr = nil
    clients.clear()
    pending_upgrades.clear()
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

proc register_task_configs*(configs_json: string) {.exportpy.} =
  let parsed = parseJson(configs_json)
  if parsed.kind != JArray:
    return
  with_state_lock:
    for item in parsed:
      if item.kind != JObject:
        continue
      let tid = item.getOrDefault("task").getStr("").strip().toLowerAscii()
      if tid.len == 0 or tid.len > max_task_name_len:
        continue
      let has_elite = item.hasKey("defaultEliteFrac") and item["defaultEliteFrac"].kind in {JFloat, JInt}
      task_configs[tid] = TaskConfig(
        is_control: item.getOrDefault("isControl").getBool(false),
        sim_phase_enabled: item.getOrDefault("simPhaseEnabled").getBool(false),
        default_auto_evolve: item.getOrDefault("defaultAutoEvolve").getBool(false),
        default_pop_size: clamp(
          item.getOrDefault("defaultPopSize").getInt(default_control_population_size),
          min_population_size,
          max_population_size,
        ),
        max_pop_size: clamp(
          item.getOrDefault("maxPopSize").getInt(control_max_population_size),
          min_population_size,
          max_population_size,
        ),
        default_min_votes: max(0, min(max_round_min_votes, item.getOrDefault("defaultMinVotes").getInt(0))),
        has_elite_override: has_elite,
        default_elite_frac: (
          if has_elite: clamp(item["defaultEliteFrac"].getFloat(0.0), min_elite_frac, max_elite_frac)
          else: default_elite_frac
        ),
      )

proc seed_room_goals*(room, goals_json: string) {.exportpy.} =
  ## Seed a room with initial PyTorch-generated goals JSON: [{"name": ..., "genomes": [...]}]
  let parsed = parseJson(goals_json)
  if parsed.kind != JArray:
    return
  with_state_lock:
    room_evolution[room] = populate_room_goals_locked(parsed, cfg.params)
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
      "upCounts": job.up_counts,
      "mutSigma": job.params.mut_sigma,
      "eliteFrac": job.params.elite_frac,
      "crossoverRate": job.params.crossover_rate,
      "populationSize": job.population_size,
      "maxVotesPerUser": job.params.max_votes_per_user,
      "minVotes": job.params.min_votes,
    }
    let meta_str = $meta
    let g_str = if job.genomes_json.len > 0: job.genomes_json else: "[]"
    var item = newStringOfCap(meta_str.len + g_str.len + 16)
    item.add(meta_str[0 ..< meta_str.len - 1])
    item.add(",\"genomes\":")
    item.add(g_str)
    item.add("}")
    parts.add(item)
  "[" & parts.join(",") & "]"

proc apply_room_init*(room, goals_json: string) {.exportpy.} =
  let parsed = parseJson(goals_json)
  if parsed.kind != JArray:
    return
  var outbox: seq[PendingBroadcast] = @[]
  with_state_lock:
    let dp =
      if room in room_evolution and room_evolution[room].default_params.max_votes_per_user > 0:
        room_evolution[room].default_params
      else:
        cfg.params
    var carry_creates = initHashSet[string]()
    var carry_tasks = initTable[string, string]()
    var carry_params = initTable[string, GoalParams]()
    if room in room_evolution and room_evolution[room].initializing:
      # keep goals requested before the room finished booting
      carry_creates = room_evolution[room].pending_creates
      carry_tasks = room_evolution[room].pending_create_tasks
      carry_params = room_evolution[room].pending_create_params
    room_evolution[room] = populate_room_goals_locked(parsed, dp)
    room_evolution[room].pending_creates = carry_creates
    room_evolution[room].pending_create_tasks = carry_tasks
    room_evolution[room].pending_create_params = carry_params
    dirty_rooms.incl(room)
    queue_evolution_state_outbox_locked(room, outbox)
  for item in outbox:
    broadcast(item.targets, item.payload)

proc apply_goal_create_impl(room, goal_name, task_override, username, genomes_json: string; min_votes: int = -1) =
  let g_arr = parseJson(genomes_json)
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
        room_evolution[room].goals[goal_name] = init_goal_population_locked(goal_name, task_name, genomes_json, g_arr.len, gp)
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

proc apply_goal_create_with_task*(room, goal_name, task_name, username, genomes_json: string; min_votes: int = -1) {.exportpy.} =
  apply_goal_create_impl(room, goal_name, task_name, username, genomes_json, min_votes)

proc clear_room_evolution*(room, goals_json, goal_name, username: string) {.exportpy, gcsafe.} =
  let parsed = parseJson(goals_json)
  if parsed.kind != JArray:
    return
  let target_goal = goal_name.strip()
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
    if target_goal.len > 0 and target_goal != "*":
      if room in room_evolution:
        var evo = room_evolution[room]
        evo.pending_creates.excl(target_goal)
        evo.pending_create_tasks.del(target_goal)
        evo.pending_create_params.del(target_goal)
        if parsed.len == 0:
          evo.goals.del(target_goal)
          var next_order: seq[string] = @[]
          for gn in evo.goal_order:
            if gn != target_goal:
              next_order.add(gn)
          evo.goal_order = next_order
          room_evolution[room] = evo
          if room != "lobby" and evo.goal_order.len == 0 and rooms.getOrDefault(room).len == 0:
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
          if target_goal in evo.goals:
            gp = evo.goals[target_goal].params
            goal_task = evo.goals[target_goal].task
          else:
            evo.goal_order.add(target_goal)
          evo.goals[target_goal] = init_goal_population_locked(target_goal, goal_task, $parsed, parsed.len, gp)
          room_evolution[room] = evo
          dirty_rooms.incl(room)
          queue_goal_updated_outbox_locked(room, evo.goals[target_goal], now_ms(), outbox)
          sys_payload = system_msg_payload(username & " cleared \"" & target_goal & "\" to gen 0")
          targets = room_targets(room)
    else:
      if room != "lobby" and parsed.len == 0:
        dirty_rooms.excl(room)
        if rooms.getOrDefault(room).len == 0:
          room_evolution.del(room)
        else:
          room_evolution[room] = RoomEvolution(
            goal_order: @[],
            goals: initTable[string, GoalPopulation](),
            pending_creates: initHashSet[string](),
            pending_create_tasks: initTable[string, string](),
            pending_create_params: initTable[string, GoalParams](),
            default_params: dp,
            initializing: false,
          )
          queue_evolution_state_outbox_locked(room, outbox)
          sys_payload = system_msg_payload(username & " cleared room evolution state")
          targets = room_targets(room)
      else:
        room_evolution[room] = populate_room_goals_locked(parsed, dp)
        dirty_rooms.incl(room)
        queue_evolution_state_outbox_locked(room, outbox)
        sys_payload = system_msg_payload(username & " cleared room evolution state to gen 0")
        targets = room_targets(room)
  for item in outbox:
    broadcast(item.targets, item.payload)
  if targets.len > 0 and sys_payload.len > 0:
    broadcast(targets, sys_payload)

proc apply_goal_reset*(room, goal_name, username, genomes_json: string) {.exportpy.} =
  let g_arr = parseJson(genomes_json)
  let fresh_lineage = merge_lineage_json("", genomes_json, 0)
  let fresh_lean = compute_lean_lineage(fresh_lineage, genomes_json, 0)
  var
    outbox: seq[PendingBroadcast] = @[]
    sys_payload: string
    targets: seq[WebSocket]
    applied = false
  with_state_lock:
    if room in room_evolution and goal_name in room_evolution[room].goals:
      let now = now_ms()
      var goal = room_evolution[room].goals[goal_name]
      goal.generation = 0
      goal.genomes_json = genomes_json
      goal.lineage_json = fresh_lineage
      goal.lean_lineage_json = fresh_lean.lean_json
      goal.lineage_total = fresh_lean.total
      goal.votes = empty_votes(max(1, g_arr.len))
      goal.phase = (if task_cfg_locked(goal.task).sim_phase_enabled: "simulating" else: "voting")
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
  let g_arr = parseJson(genomes_json)
  var
    outbox: seq[PendingBroadcast] = @[]
    sys_payload: string
    targets: seq[WebSocket]
    applied = false
  with_state_lock:
    if room in room_evolution and goal_name in room_evolution[room].goals:
      let now = now_ms()
      var goal = room_evolution[room].goals[goal_name]
      inc goal.generation
      goal.genomes_json = genomes_json
      goal.lineage_json = merge_lineage_json(goal.lineage_json, genomes_json, goal.generation)
      goal.refresh_goal_lineage_cache()
      goal.votes = empty_votes(max(1, g_arr.len))
      goal.phase = (if task_cfg_locked(goal.task).sim_phase_enabled: "simulating" else: "voting")
      goal.sim_completed = 0
      goal.countdown_ends_at_ms = 0
      goal.evolving = false
      goal.evolving_started_at_ms = 0
      room_evolution[room].goals[goal_name] = goal
      dirty_rooms.incl(room)
      queue_goal_updated_outbox_locked(room, goal, now, outbox)
      sys_payload = system_msg_payload(
        "Crowd evolved \"" & goal_name & "\" -> gen " & $goal.generation & " (top score " & $top_score & ")"
      )
      targets = room_targets(room)
      applied = true
  if applied:
    for item in outbox:
      broadcast(item.targets, item.payload)
    if targets.len > 0:
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

proc apply_goal_resize*(room, goal_name, username, genomes_json: string) {.exportpy.} =
  let g_len = max(1, parseJson(genomes_json).len)
  var
    outbox: seq[PendingBroadcast] = @[]
    sys_payload: string
    targets: seq[WebSocket]
    applied = false
  with_state_lock:
    if room in room_evolution and goal_name in room_evolution[room].goals:
      let now = now_ms()
      var goal = room_evolution[room].goals[goal_name]
      var next_votes = empty_votes(g_len)
      let keep = min(goal.votes.len, g_len)
      for i in 0 ..< keep:
        next_votes[i] = goal.votes[i]
      goal.genomes_json = genomes_json
      goal.lineage_json = merge_missing_lineage_json(goal.lineage_json, genomes_json, goal.generation)
      goal.refresh_goal_lineage_cache()
      goal.votes = next_votes
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
  let parsed = parseJson(messages_json)
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
  var arr = newJArray()
  for item in batch:
    arr.add(%*{
      "room": item.room,
      "username": item.username,
      "message": item.message,
    })
  $arr

proc get_room_goal_pop_sizes*(room: string): string {.exportpy.} =
  ## Return JSON object mapping active goal names in `room` to their population sizes.
  var obj = newJObject()
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
  var arr = newJArray()
  with_state_lock:
    if dirty_rooms.len == 0:
      return "[]"
    for room in dirty_rooms:
      if room in room_evolution and not room_evolution[room].initializing:
        arr.add(%*{
          "room": room,
          "stateJson": serialize_room_evolution(room_evolution[room]),
        })
    dirty_rooms.clear()
  $arr

proc restore_room_state*(room, state_json: string) {.exportpy.} =
  if state_json.len == 0:
    return
  var parsed: JsonNode
  try:
    parsed = parseJson(state_json)
  except CatchableError:
    return
  if parsed.kind != JObject:
    return
  let goals_node = parsed.getOrDefault("goals")
  if goals_node == nil or goals_node.kind != JArray:
    return

  var outbox: seq[PendingBroadcast] = @[]
  with_state_lock:
    let dp = parse_goal_params_json(parsed.getOrDefault("defaultParams"), cfg.params)
    var evo = RoomEvolution(
      goal_order: @[],
      goals: initTable[string, GoalPopulation](),
      pending_creates: initHashSet[string](),
      pending_create_tasks: initTable[string, string](),
      pending_create_params: initTable[string, GoalParams](),
      default_params: dp,
      initializing: false,
    )
    for item in goals_node:
      if item.kind != JObject:
        continue
      let name = sanitize_goal_name(item.getOrDefault("name").getStr(""))
      let task = sanitize_task_name_locked(item.getOrDefault("task").getStr("image"))
      let genomes_arr = item.getOrDefault("genomes")
      if name.len == 0 or genomes_arr == nil or genomes_arr.kind != JArray or genomes_arr.len == 0:
        continue
      let gen = max(0, item.getOrDefault("generation").getInt(0))
      var gp = parse_goal_params_json(item.getOrDefault("params"), dp)
      if gp.min_votes <= 0:
        gp.min_votes = default_min_votes_for_task_locked(task)
      clamp_control_min_votes_locked(task, gp)
      let g_json = $genomes_arr
      let lin_node = item.getOrDefault("lineage")
      let l_json =
        if lin_node != nil and lin_node.kind == JArray and lin_node.len > 0:
          $lin_node
        else:
          $extract_lineage_nodes(g_json, gen)
      let lean = compute_lean_lineage(l_json, g_json, gen)
      let tc = task_cfg_locked(task)
      let default_phase = if tc.sim_phase_enabled: "simulating" else: "voting"
      let raw_phase = item.getOrDefault("phase").getStr(default_phase)
      let eff_phase = if raw_phase in ["simulating", "voting"]: raw_phase else: default_phase
      let eff_sim_done = clamp(item.getOrDefault("simCompleted").getInt(0), 0, max(1, genomes_arr.len))
      let eff_auto = item.getOrDefault("autoEvolve").getBool(tc.default_auto_evolve)
      evo.goal_order.add(name)
      evo.goals[name] = GoalPopulation(
        name: name,
        task: task,
        generation: gen,
        genomes_json: g_json,
        lineage_json: l_json,
        lean_lineage_json: lean.lean_json,
        lineage_total: lean.total,
        votes: empty_votes(max(1, genomes_arr.len)),
        phase: eff_phase,
        sim_completed: eff_sim_done,
        auto_evolve: eff_auto,
        countdown_ends_at_ms: 0,
        evolving: false,
        evolving_started_at_ms: 0,
        params: gp,
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
    rooms.clear()
    room_history.clear()
    room_evolution.clear()
    dirty_rooms.clear()
    pending_saves = @[]
    pending_evo_jobs = @[]
