/* Compact promise-free wrapper around the training exports. */
export function createAgentApi(Module) {
  const call = (name, returnType, argTypes, args) =>
    Module.ccall(name, returnType, argTypes, args);
  const json = name => JSON.parse(call(name, "string", [], []));
  return {
    version: () => call("agent_api_version", "number", [], []),
    reset: (skill = 2, episode = 1, map = 1, seed = 1) =>
      !!call("agent_reset", "number", ["number", "number", "number", "number"], [skill, episode, map, seed]),
    action: ({ forward = 0, strafe = 0, turn = 0, fire = false, use = false }) =>
      call("agent_set_action", "number", ["number", "number", "number", "number"],
        [forward, strafe, turn, (fire ? 1 : 0) | (use ? 2 : 0)]),
    step: (tics = 1) => call("agent_step", "number", ["number"], [tics]),
    render: () => call("agent_render", "number", [], []),
    pause: () => call("agent_pause", "number", [], []),
    resume: () => call("agent_resume", "number", [], []),
    state: () => json("agent_state_json"),
    probeMove: (forward, strafe) => !!call("agent_probe_move", "number", ["number", "number"], [forward, strafe]),
    map: () => Array.from({ length: call("agent_map_line_count", "number", [], []) }, (_, index) =>
      JSON.parse(call("agent_map_line_json", "string", ["number"], [index]))),
    enemies: () => Array.from({ length: call("agent_enemy_count", "number", [], []) }, (_, index) =>
      JSON.parse(call("agent_enemy_json", "string", ["number"], [index])))
  };
}
