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
  max_room_name_len = 64

  min_population_size = 3
  max_population_size = 64
  default_population_size = 30

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

  default_min_votes = 15
  default_countdown_ms = 8000'i64
  vote_burst_window_ms = 1000'i64
  vote_burst_max_ops = 64
  max_lineage_nodes = 300
  ticker_sleep_ms = 40
  # Large JSON payloads compress ~3x; below this it is not worth the CPU.
  gzip_min_bytes = 4096

type
  EvoJobKind {.pure.} = enum
    init_room = "init_room"
    create = "create"
    evolve = "evolve"
    resize = "resize"

  GoalParams = object
    max_votes_per_user: int
    mut_sigma: float
    elite_frac: float
    crossover_rate: float

  RuntimeConfig = object
    min_votes: int
    min_votes_explicit: bool
    countdown_ms: int64
    population_size: int
    params: GoalParams

  PendingClient = object
    room: string
    username: string
    initial_goal: string
    initial_pop_size: int
    initial_params: GoalParams
    gzip: bool

  ClientInfo = object
    ws: WebSocket
    room: string
    username: string
    vote_timestamps_ms: seq[int64]
    gzip: bool

  PendingPersist = object
    room: string
    username: string
    message: string

  GoalPopulation = object
    name: string
    generation: int
    genomes_json: string
    lineage_json: string
    ## Per-candidate vote map: WebSocket -> +1 (up) or -1 (down).
    votes: seq[Table[WebSocket, int]]
    countdown_ends_at_ms: int64
    evolving: bool
    params: GoalParams

  RoomEvolution = object
    goal_order: seq[string]
    goals: Table[string, GoalPopulation]
    pending_creates: HashSet[string]
    pending_create_params: Table[string, GoalParams]
    default_params: GoalParams
    initializing: bool

  EvoJob = object
    kind: EvoJobKind
    room: string
    goal: string
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
    mut_sigma: default_mut_sigma,
    elite_frac: default_elite_frac,
    crossover_rate: default_crossover_rate,
  )

func default_runtime_config(): RuntimeConfig {.inline.} =
  RuntimeConfig(
    min_votes: default_min_votes,
    min_votes_explicit: false,
    countdown_ms: default_countdown_ms,
    population_size: default_population_size,
    params: default_goal_params(),
  )

var
  state_lock: Lock
  ready_lock: Lock
  ready_cond: Cond

  cfg = default_runtime_config()
  clients: Table[WebSocket, ClientInfo]
  pending_upgrades: Table[WebSocket, PendingClient]
  rooms: Table[string, HashSet[WebSocket]]
  room_history: Table[string, seq[string]]
  room_evolution: Table[string, RoomEvolution]
  dirty_rooms: HashSet[string]
  pending_saves: seq[PendingPersist]
  pending_evo_jobs: seq[EvoJob]
  admin_password = ""

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

func clamp_goal_params(p: GoalParams): GoalParams {.inline.} =
  GoalParams(
    max_votes_per_user: clamp(p.max_votes_per_user, min_votes_per_user, max_votes_per_user),
    mut_sigma: clamp(p.mut_sigma, min_mut_sigma, max_mut_sigma),
    elite_frac: clamp(p.elite_frac, min_elite_frac, max_elite_frac),
    crossover_rate: clamp(p.crossover_rate, min_crossover_rate, max_crossover_rate),
  )

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
  if c.min_votes_explicit:
    return max(1, c.min_votes)
  let n = max(1, goal.votes.len)
  clamp((n + 1) div 2, 1, n)

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
  if goal.evolving:
    return
  let counted = count_goal_votes(goal)
  let req = effective_min_votes(goal, c)
  let ready =
    if c.min_votes_explicit:
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
    result.add(%*{
      "id": gid,
      "gen": gen,
      "parents": parents_node,
      "op": op_str,
      "weights": item.getOrDefault("weights"),
      "biases": item.getOrDefault("biases"),
      "activations": item.getOrDefault("activations"),
    })

proc lineage_payload_node(goal: GoalPopulation; full: bool): tuple[lineage: JsonNode, total: int] {.gcsafe.} =
  ## Sends only the current generation's ancestry by default; the client asks
  ## for the archived `all` view on demand. Live nodes resolve their weights
  ## from `goal.genomes`, so omit that duplicate data.
  result.lineage = newJArray()
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

  var by_id = initTable[int, JsonNode]()
  for node in parsed:
    by_id[node.getOrDefault("id").getInt(0)] = node

  var keep = initHashSet[int]()
  var queue: seq[int] = @[]
  for node in parsed:
    if node.getOrDefault("gen").getInt(-1) == goal.generation:
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

  for node in parsed:
    let gid = node.getOrDefault("id").getInt(0)
    if gid notin keep:
      continue
    if node.getOrDefault("gen").getInt(-1) == goal.generation:
      var lean = node
      lean.delete("weights")
      lean.delete("biases")
      lean.delete("activations")
      result.lineage.add(lean)
    else:
      result.lineage.add(node)

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
) {.gcsafe.} =
  let vinfo = build_votes_json(goal, client_map)
  obj["votes"] = vinfo.votes_arr
  obj["totalVotes"] = %vinfo.total_votes
  obj["votedCandidates"] = %vinfo.voted_candidates
  obj["likedCandidates"] = %vinfo.liked_candidates
  obj["minVotes"] = %effective_min_votes(goal, c)
  obj["maxVotesPerUser"] = %goal.params.max_votes_per_user
  obj["countdownMs"] = %remaining_countdown_ms(goal, now)
  obj["countdownDurationMs"] = %c.countdown_ms
  obj["evolving"] = %goal.evolving
  obj["mutationSigma"] = %goal.params.mut_sigma
  obj["eliteFrac"] = %goal.params.elite_frac
  obj["crossoverRate"] = %goal.params.crossover_rate
  obj["populationSize"] = %goal.votes.len

proc goal_to_json(
  goal: GoalPopulation;
  c: RuntimeConfig;
  client_map: Table[WebSocket, ClientInfo];
  now: int64;
  full_lineage = false;
): JsonNode {.gcsafe.} =
  let genomes_node =
    if goal.genomes_json.len > 0: parseJson(goal.genomes_json)
    else: newJArray()
  let lineage_payload = lineage_payload_node(goal, full_lineage)
  result = %*{
    "name": goal.name,
    "generation": goal.generation,
    "genomes": genomes_node,
    "lineage": lineage_payload.lineage,
    "lineageTotal": %lineage_payload.total,
  }
  result.attach_round_fields(goal, c, client_map, now)

proc build_votes_updated_payload(
  goal: GoalPopulation;
  c: RuntimeConfig;
  client_map: Table[WebSocket, ClientInfo];
  now: int64;
): string {.gcsafe.} =
  let node = %*{
    "type": "votes_updated",
    "goal": goal.name,
  }
  node.attach_round_fields(goal, c, client_map, now)
  $node

proc build_goal_updated_payload(
  goal: GoalPopulation;
  c: RuntimeConfig;
  client_map: Table[WebSocket, ClientInfo];
  now: int64;
): string {.gcsafe.} =
  $(%*{
    "type": "goal_updated",
    "goal": goal_to_json(goal, c, client_map, now),
  })

proc build_evolution_state_payload(room: string): string {.gcsafe.} =
  let now = now_ms()
  {.gcsafe.}:
    var goals_arr = newJArray()
    if room in room_evolution:
      let evo = room_evolution[room]
      for name in evo.goal_order:
        if name in evo.goals:
          goals_arr.add(goal_to_json(evo.goals[name], cfg, clients, now))
    $(%*{
      "type": "evolution_state",
      "goals": goals_arr,
    })

proc goal_params_json(p: GoalParams): JsonNode {.gcsafe.} =
  %*{
    "maxVotesPerUser": p.max_votes_per_user,
    "mutSigma": p.mut_sigma,
    "eliteFrac": p.elite_frac,
    "crossoverRate": p.crossover_rate,
  }

proc parse_goal_params_json(node: JsonNode; fallback: GoalParams): GoalParams {.gcsafe.} =
  if node == nil or node.kind != JObject:
    return fallback
  clamp_goal_params(GoalParams(
    max_votes_per_user: node.getOrDefault("maxVotesPerUser").getInt(fallback.max_votes_per_user),
    mut_sigma: node.getOrDefault("mutSigma").getFloat(fallback.mut_sigma),
    elite_frac: node.getOrDefault("eliteFrac").getFloat(fallback.elite_frac),
    crossover_rate: node.getOrDefault("crossoverRate").getFloat(fallback.crossover_rate),
  ))

proc serialize_room_evolution(evo: RoomEvolution): string {.gcsafe.} =
  var goals_arr = newJArray()
  for name in evo.goal_order:
    if name notin evo.goals:
      continue
    let g = evo.goals[name]
    let genomes_node =
      if g.genomes_json.len > 0: parseJson(g.genomes_json)
      else: newJArray()
    let lineage_node =
      if g.lineage_json.len > 0: parseJson(g.lineage_json)
      else: extract_lineage_nodes(g.genomes_json, g.generation)
    goals_arr.add(%*{
      "name": g.name,
      "generation": g.generation,
      "genomes": genomes_node,
      "lineage": lineage_node,
      "params": goal_params_json(g.params),
    })
  $(%*{
    "defaultParams": goal_params_json(evo.default_params),
    "goals": goals_arr,
  })

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

proc client_uses_gzip(ws: WebSocket): bool {.gcsafe.} =
  with_state_lock:
    result = ws in clients and clients[ws].gzip

proc compress_payload(payload: string): string {.gcsafe.} =
  try:
    result = compress(payload, BestSpeed, dfGzip)
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

proc init_goal_population(
  name, genomes_json: string;
  pop_len: int;
  gp: GoalParams;
): GoalPopulation {.gcsafe.} =
  GoalPopulation(
    name: name,
    generation: 0,
    genomes_json: genomes_json,
    lineage_json: merge_lineage_json("", genomes_json, 0),
    votes: empty_votes(max(1, pop_len)),
    countdown_ends_at_ms: 0,
    evolving: false,
    params: gp,
  )

proc populate_room_goals(
  parsed: JsonNode;
  dp: GoalParams;
): RoomEvolution {.gcsafe.} =
  result = RoomEvolution(
    goal_order: @[],
    goals: initTable[string, GoalPopulation](),
    pending_creates: initHashSet[string](),
    pending_create_params: initTable[string, GoalParams](),
    default_params: dp,
    initializing: false,
  )
  for item in parsed:
    let name = item["name"].getStr()
    let g_arr = item["genomes"]
    let g_json = $g_arr
    result.goal_order.add(name)
    result.goals[name] = init_goal_population(name, g_json, g_arr.len, dp)

proc upgrade_handler(request: Request) {.gcsafe.} =
  let room = request.pathParams["room"]
  if not is_valid_room(room):
    request.respond(400, body = "Invalid room name")
    return

  let username = sanitize_username(request.queryParams["username"])
  let initial_goal = sanitize_goal_name(request.queryParams["goal"])
  let wants_gzip = request.queryParams["gzip"] == "1"
  var base_cfg: RuntimeConfig
  with_state_lock:
    base_cfg = cfg

  let req_pop = parse_clamped_int(
    request.queryParams["pop"],
    base_cfg.population_size, min_population_size, max_population_size,
  )
  let req_params = GoalParams(
    max_votes_per_user: parse_clamped_int(
      request.queryParams["max_votes"],
      base_cfg.params.max_votes_per_user, min_votes_per_user, max_votes_per_user,
    ),
    mut_sigma: parse_clamped_float(
      request.queryParams["sigma"],
      base_cfg.params.mut_sigma, min_mut_sigma, max_mut_sigma,
    ),
    elite_frac: parse_clamped_float(
      request.queryParams["elite_frac"],
      base_cfg.params.elite_frac, min_elite_frac, max_elite_frac,
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
      initial_goal: initial_goal,
      initial_pop_size: req_pop,
      initial_params: req_params,
      gzip: wants_gzip,
    )

proc on_open(ws: WebSocket) {.gcsafe.} =
  var
    identity_payload: string
    history_payload: string
    evolution_payload = ""
    presence_payload: string
    join_payload: string
    targets: seq[WebSocket]
    found = false
    has_evolution_ready = false

  with_state_lock:
    if ws in pending_upgrades:
      let info = pending_upgrades[ws]
      pending_upgrades.del(ws)
      let unique_name = make_unique_username(info.room, info.username)
      clients[ws] = ClientInfo(
        ws: ws,
        room: info.room,
        username: unique_name,
        gzip: info.gzip,
      )
      if info.room notin rooms:
        rooms[info.room] = initHashSet[WebSocket]()
      rooms[info.room].incl(ws)

      identity_payload = $(%*{"type": "identity", "username": unique_name})
      history_payload = build_history_payload(info.room)

      if info.room in room_evolution and not room_evolution[info.room].initializing:
        evolution_payload = build_evolution_state_payload(info.room)
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
          pending_create_params: initTable[string, GoalParams](),
          default_params: room_params,
          initializing: true,
        )
        pending_evo_jobs.add(EvoJob(
          kind: EvoJobKind.init_room,
          room: info.room,
          goal: (if is_private: info.initial_goal else: ""),
          username: unique_name,
          population_size: room_pop,
          params: room_params,
        ))

      presence_payload = build_presence_payload(info.room)
      join_payload = system_msg_payload(unique_name & " joined")
      targets = room_targets(info.room)
      found = true

  if found:
    send_payload(ws, identity_payload)
    send_payload(ws, history_payload)
    if has_evolution_ready:
      send_payload(ws, evolution_payload)
    broadcast(targets, presence_payload)
    broadcast(targets, join_payload)

proc handle_rename(ws: WebSocket; payload: JsonNode) {.gcsafe.} =
  let desired_name = sanitize_username(payload.getOrDefault("username").getStr(""))
  var
    old_name = ""
    assigned_name = ""
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
      let room = clients[ws].room
      assigned_name = make_unique_username(room, desired_name, ws, has_exclude = true)
      if assigned_name != old_name:
        clients[ws].username = assigned_name
        let now = now_ms()
        presence_payload = build_presence_payload(room)
        rename_payload = system_msg_payload(old_name & " is now " & assigned_name)
        cursor_leave_payload = $(%*{"type": "cursor_leave", "username": old_name})
        if room in room_evolution:
          for g_name in room_evolution[room].goal_order:
            let goal = room_evolution[room].goals[g_name]
            var has_user_vote = false
            for i in 0 ..< goal.votes.len:
              if ws in goal.votes[i]:
                has_user_vote = true
                break
            if has_user_vote:
              vote_payloads.add(build_votes_updated_payload(goal, cfg, clients, now))
        targets = room_targets(room)
        other_targets = room_targets(room, ws, has_exclude = true)
        changed = true

  if changed:
    send_payload(ws, $(%*{"type": "identity", "username": assigned_name}))
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

  with_state_lock:
    if ws in clients:
      let info = clients[ws]
      if info.room != "lobby" and
        info.room in room_evolution and
        goal_name notin room_evolution[info.room].goals and
        goal_name notin room_evolution[info.room].pending_creates:
        let d = room_evolution[info.room].default_params
        let req_pop = clamp(
          payload.getOrDefault("population_size").getInt(cfg.population_size),
          min_population_size, max_population_size,
        )
        let g_params = clamp_goal_params(GoalParams(
          max_votes_per_user: payload.getOrDefault("max_votes_per_user").getInt(d.max_votes_per_user),
          mut_sigma: payload.getOrDefault("mutation_sigma").getFloat(d.mut_sigma),
          elite_frac: payload.getOrDefault("elite_frac").getFloat(d.elite_frac),
          crossover_rate: payload.getOrDefault("crossover_rate").getFloat(d.crossover_rate),
        ))
        room_evolution[info.room].pending_creates.incl(goal_name)
        room_evolution[info.room].pending_create_params[goal_name] = g_params
        pending_evo_jobs.add(EvoJob(
          kind: EvoJobKind.create,
          room: info.room,
          goal: goal_name,
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
        if not goal.evolving and idx < goal.votes.len and (not has_req_gen or req_gen == goal.generation):
          let now = now_ms()
          let prev_vote = goal.votes[idx].getOrDefault(ws, 0)
          let next_vote = if prev_vote == req_vote: 0 else: req_vote
          let cap = goal.params.max_votes_per_user
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
            votes_payload = build_votes_updated_payload(goal, cfg, clients, now)
            targets = room_targets(client_info.room)
            updated = true

  if throttle_payload.len > 0:
    send_payload(ws, throttle_payload)
  if updated:
    broadcast(targets, votes_payload)

proc on_close(ws: WebSocket) {.gcsafe.} =
  var
    left_payload: string
    presence_payload: string
    cursor_leave_payload: string
    vote_payloads: seq[string] = @[]
    targets: seq[WebSocket]
    had_client = false

  with_state_lock:
    pending_upgrades.del(ws)
    if ws in clients:
      let info = clients[ws]
      let now = now_ms()
      clients.del(ws)
      if info.room in rooms:
        rooms[info.room].excl(ws)
        if rooms[info.room].len == 0:
          rooms.del(info.room)
      if info.room in room_evolution:
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
            vote_payloads.add(build_votes_updated_payload(goal, cfg, clients, now))

      left_payload = system_msg_payload(info.username & " left")
      cursor_leave_payload = $(%*{"type": "cursor_leave", "username": info.username})
      presence_payload = build_presence_payload(info.room)
      targets = room_targets(info.room)
      had_client = true

  if had_client:
    broadcast(targets, cursor_leave_payload)
    broadcast(targets, left_payload)
    broadcast(targets, presence_payload)
    for vp in vote_payloads:
      broadcast(targets, vp)

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

proc websocket_handler(
  ws: WebSocket;
  event: WebSocketEvent;
  message: Message;
) {.gcsafe.} =
  case event
  of OpenEvent:
    on_open(ws)
  of MessageEvent:
    if message.kind != TextMessage:
      return
    var payload: JsonNode
    try:
      payload = parseJson(message.data)
    except CatchableError:
      return
    if payload.kind != JObject:
      return
    case payload.getOrDefault("type").getStr("chat")
    of "rename": handle_rename(ws, payload)
    of "cursor": handle_cursor(ws, payload)
    of "chat": handle_chat(ws, payload)
    of "create_goal": handle_create_goal(ws, payload)
    of "cast_vote": handle_cast_vote(ws, payload)
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
      let now = now_ms()

      with_state_lock:
        for room, evo in room_evolution.mpairs:
          for g_name in evo.goal_order:
            if g_name notin evo.goals:
              continue
            var goal = evo.goals[g_name]
            if not goal.evolving and goal.countdown_ends_at_ms > 0 and now >= goal.countdown_ends_at_ms:
              goal.evolving = true
              goal.countdown_ends_at_ms = 0
              evo.goals[g_name] = goal

              let counted = count_goal_votes(goal)
              let pop_len = goal.votes.len
              var scores = newSeq[int](pop_len)
              var up_counts = newSeq[int](pop_len)
              for i in 0 ..< pop_len:
                scores[i] = counted.stats[i].score
                up_counts[i] = counted.stats[i].up

              pending_evo_jobs.add(EvoJob(
                kind: EvoJobKind.evolve,
                room: room,
                goal: g_name,
                genomes_json: goal.genomes_json,
                scores: scores,
                up_counts: up_counts,
                population_size: pop_len,
                params: goal.params,
              ))

              let targets = room_targets(room)
              if targets.len > 0:
                outbox.add(PendingBroadcast(
                  targets: targets,
                  payload: build_votes_updated_payload(goal, cfg, clients, now),
                ))

      for item in outbox:
        broadcast(item.targets, item.payload)

proc constant_time_eq(a, b: string): bool {.gcsafe.} =
  if a.len != b.len:
    return false
  var diff = 0'u8
  for i in 0 ..< a.len:
    diff = diff or (uint8(a[i]) xor uint8(b[i]))
  diff == 0'u8

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

proc serve_thread_proc(args: tuple[host: string, port: int]) {.thread.} =
  {.gcsafe.}:
    var srv: Server = nil
    try:
      var router: Router
      router.get("/ws/chat/@room", upgrade_handler)
      router.get("/ws/chat/@room/", upgrade_handler)
      router.get("/internal/room-goals/@room", internal_room_goals_handler)
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

proc set_round_config*(min_votes, countdown_ms: int) {.exportpy.} =
  with_state_lock:
    cfg.min_votes = max(1, min_votes)
    cfg.min_votes_explicit = true
    cfg.countdown_ms = max(50'i64, int64(countdown_ms))

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

proc seed_room_goals*(room, goals_json: string) {.exportpy.} =
  ## Seed a room with initial PyTorch-generated goals JSON: [{"name": ..., "genomes": [...]}]
  let parsed = parseJson(goals_json)
  if parsed.kind != JArray:
    return
  with_state_lock:
    room_evolution[room] = populate_room_goals(parsed, cfg.params)
    dirty_rooms.incl(room)

proc drain_evo_jobs*(): string {.exportpy.} =
  var jobs: seq[EvoJob]
  with_state_lock:
    jobs = move pending_evo_jobs
    pending_evo_jobs = @[]
  var arr = newJArray()
  for job in jobs:
    arr.add(%*{
      "kind": $job.kind,
      "room": job.room,
      "goal": job.goal,
      "username": job.username,
      "genomes": (if job.genomes_json.len > 0: parseJson(job.genomes_json) else: newJArray()),
      "scores": job.scores,
      "upCounts": job.up_counts,
      "mutSigma": job.params.mut_sigma,
      "eliteFrac": job.params.elite_frac,
      "crossoverRate": job.params.crossover_rate,
      "populationSize": job.population_size,
      "maxVotesPerUser": job.params.max_votes_per_user,
    })
  $arr

proc apply_room_init*(room, goals_json: string) {.exportpy.} =
  let parsed = parseJson(goals_json)
  if parsed.kind != JArray:
    return
  var
    evo_payload: string
    targets: seq[WebSocket]
  with_state_lock:
    let dp =
      if room in room_evolution and room_evolution[room].default_params.max_votes_per_user > 0:
        room_evolution[room].default_params
      else:
        cfg.params
    var carry_creates = initHashSet[string]()
    var carry_params = initTable[string, GoalParams]()
    if room in room_evolution and room_evolution[room].initializing:
      # keep goals requested before the room finished booting
      carry_creates = room_evolution[room].pending_creates
      carry_params = room_evolution[room].pending_create_params
    room_evolution[room] = populate_room_goals(parsed, dp)
    room_evolution[room].pending_creates = carry_creates
    room_evolution[room].pending_create_params = carry_params
    dirty_rooms.incl(room)
    evo_payload = build_evolution_state_payload(room)
    targets = room_targets(room)
  if targets.len > 0:
    broadcast(targets, evo_payload)

proc apply_goal_create*(room, goal_name, username, genomes_json: string) {.exportpy.} =
  let g_arr = parseJson(genomes_json)
  var
    evo_payload: string
    sys_payload: string
    targets: seq[WebSocket]
    applied = false
  with_state_lock:
    if room in room_evolution:
      room_evolution[room].pending_creates.excl(goal_name)
      var gp = room_evolution[room].default_params
      if gp.max_votes_per_user <= 0:
        gp = cfg.params
      if goal_name in room_evolution[room].pending_create_params:
        gp = room_evolution[room].pending_create_params[goal_name]
        room_evolution[room].pending_create_params.del(goal_name)
      if goal_name notin room_evolution[room].goals:
        room_evolution[room].goal_order.add(goal_name)
        room_evolution[room].goals[goal_name] = init_goal_population(goal_name, genomes_json, g_arr.len, gp)
        dirty_rooms.incl(room)
        evo_payload = build_evolution_state_payload(room)
        sys_payload = system_msg_payload(username & " created goal \"" & goal_name & "\"")
        targets = room_targets(room)
        applied = true
  if applied and targets.len > 0:
    broadcast(targets, evo_payload)
    broadcast(targets, sys_payload)

proc clear_room_evolution*(room, goals_json, goal_name, username: string) {.exportpy, gcsafe.} =
  let parsed = parseJson(goals_json)
  if parsed.kind != JArray:
    return
  let target_goal = goal_name.strip()
  var
    out_payload = ""
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
            out_payload = build_evolution_state_payload(room)
            sys_payload = system_msg_payload(username & " cleared \"" & target_goal & "\"")
            targets = room_targets(room)
        else:
          var gp = dp
          if target_goal in evo.goals:
            gp = evo.goals[target_goal].params
          else:
            evo.goal_order.add(target_goal)
          evo.goals[target_goal] = init_goal_population(target_goal, $parsed, parsed.len, gp)
          room_evolution[room] = evo
          dirty_rooms.incl(room)
          out_payload = build_goal_updated_payload(evo.goals[target_goal], cfg, clients, now_ms())
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
            pending_create_params: initTable[string, GoalParams](),
            default_params: dp,
            initializing: false,
          )
          out_payload = build_evolution_state_payload(room)
          sys_payload = system_msg_payload(username & " cleared room evolution state")
          targets = room_targets(room)
      else:
        room_evolution[room] = populate_room_goals(parsed, dp)
        dirty_rooms.incl(room)
        out_payload = build_evolution_state_payload(room)
        sys_payload = system_msg_payload(username & " cleared room evolution state to gen 0")
        targets = room_targets(room)
  if targets.len > 0 and out_payload.len > 0:
    broadcast(targets, out_payload)
    broadcast(targets, sys_payload)

proc apply_goal_reset*(room, goal_name, username, genomes_json: string) {.exportpy.} =
  let g_arr = parseJson(genomes_json)
  var
    goal_payload: string
    sys_payload: string
    targets: seq[WebSocket]
    applied = false
  with_state_lock:
    if room in room_evolution and goal_name in room_evolution[room].goals:
      let now = now_ms()
      var goal = room_evolution[room].goals[goal_name]
      goal.generation = 0
      goal.genomes_json = genomes_json
      goal.lineage_json = merge_lineage_json("", genomes_json, 0)
      goal.votes = empty_votes(max(1, g_arr.len))
      goal.countdown_ends_at_ms = 0
      goal.evolving = false
      room_evolution[room].goals[goal_name] = goal
      dirty_rooms.incl(room)
      goal_payload = build_goal_updated_payload(goal, cfg, clients, now)
      sys_payload = system_msg_payload(username & " reset \"" & goal_name & "\" to gen 0")
      targets = room_targets(room)
      applied = true
  if applied and targets.len > 0:
    broadcast(targets, goal_payload)
    broadcast(targets, sys_payload)

proc apply_goal_evolve*(room, goal_name, genomes_json: string; top_score: int) {.exportpy.} =
  let g_arr = parseJson(genomes_json)
  var
    goal_payload: string
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
      goal.votes = empty_votes(max(1, g_arr.len))
      goal.countdown_ends_at_ms = 0
      goal.evolving = false
      room_evolution[room].goals[goal_name] = goal
      dirty_rooms.incl(room)
      goal_payload = build_goal_updated_payload(goal, cfg, clients, now)
      sys_payload = system_msg_payload(
        "Crowd evolved \"" & goal_name & "\" -> gen " & $goal.generation & " (top score " & $top_score & ")"
      )
      targets = room_targets(room)
      applied = true
  if applied and targets.len > 0:
    broadcast(targets, goal_payload)
    broadcast(targets, sys_payload)

proc apply_goal_resize*(room, goal_name, username, genomes_json: string) {.exportpy.} =
  let g_len = max(1, parseJson(genomes_json).len)
  var
    goal_payload: string
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
      goal.votes = next_votes
      goal.evolving = false
      goal.sync_goal_countdown(cfg, now)
      room_evolution[room].goals[goal_name] = goal
      dirty_rooms.incl(room)
      goal_payload = build_goal_updated_payload(goal, cfg, clients, now)
      sys_payload = system_msg_payload(username & " set \"" & goal_name & "\" population to " & $g_len)
      targets = room_targets(room)
      applied = true
  if applied and targets.len > 0:
    broadcast(targets, goal_payload)
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

  var
    evo_payload = ""
    targets: seq[WebSocket] = @[]
  with_state_lock:
    let dp = parse_goal_params_json(parsed.getOrDefault("defaultParams"), cfg.params)
    var evo = RoomEvolution(
      goal_order: @[],
      goals: initTable[string, GoalPopulation](),
      pending_creates: initHashSet[string](),
      pending_create_params: initTable[string, GoalParams](),
      default_params: dp,
      initializing: false,
    )
    for item in goals_node:
      if item.kind != JObject:
        continue
      let name = sanitize_goal_name(item.getOrDefault("name").getStr(""))
      let genomes_arr = item.getOrDefault("genomes")
      if name.len == 0 or genomes_arr == nil or genomes_arr.kind != JArray or genomes_arr.len == 0:
        continue
      let gen = max(0, item.getOrDefault("generation").getInt(0))
      let gp = parse_goal_params_json(item.getOrDefault("params"), dp)
      let g_json = $genomes_arr
      let lin_node = item.getOrDefault("lineage")
      let l_json =
        if lin_node != nil and lin_node.kind == JArray and lin_node.len > 0:
          $lin_node
        else:
          $extract_lineage_nodes(g_json, gen)
      evo.goal_order.add(name)
      evo.goals[name] = GoalPopulation(
        name: name,
        generation: gen,
        genomes_json: g_json,
        lineage_json: l_json,
        votes: empty_votes(max(1, genomes_arr.len)),
        countdown_ends_at_ms: 0,
        evolving: false,
        params: gp,
      )
    if evo.goal_order.len > 0 or room != "lobby":
      room_evolution[room] = evo
      dirty_rooms.excl(room)
      if room in rooms and rooms[room].len > 0:
        evo_payload = build_evolution_state_payload(room)
        targets = room_targets(room)
  if evo_payload.len > 0 and targets.len > 0:
    broadcast(targets, evo_payload)

proc set_admin_password*(pw: string) {.exportpy.} =
  with_state_lock:
    admin_password = pw

proc clear_state*() {.exportpy.} =
  with_state_lock:
    cfg = default_runtime_config()
    admin_password = ""
    clients.clear()
    pending_upgrades.clear()
    rooms.clear()
    room_history.clear()
    room_evolution.clear()
    dirty_rooms.clear()
    pending_saves = @[]
    pending_evo_jobs = @[]
