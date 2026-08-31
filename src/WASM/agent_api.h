#ifndef DWASM_AGENT_API_H
#define DWASM_AGENT_API_H

#include "d_ticcmd.h"

#ifdef __cplusplus
extern "C" {
#endif

int agent_api_version(void);
int agent_reset(int skill, int episode, int map, unsigned int seed);
int agent_set_action(int forward, int strafe, int turn, int buttons);
int agent_step(int tics);
int agent_render(void);
int agent_pause(void);
int agent_resume(void);
void agent_override_ticcmd(ticcmd_t *cmd);
void agent_teacher_begin(unsigned int seed, int max_tics, int capture_every);
void agent_teacher_after_render(void);
int agent_teacher_active(void);
const char *agent_state_json(void);
int agent_probe_move(int forward_units, int strafe_units);
int agent_map_line_count(void);
const char *agent_map_line_json(int index);
int agent_enemy_count(void);
const char *agent_enemy_json(int index);

#ifdef __cplusplus
}
#endif
#endif
