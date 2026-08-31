/* Deterministic, training-only interface for the Emscripten build. */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "agent_api.h"
#include "doomdef.h"
#include "doomstat.h"
#include "d_event.h"
#include "g_game.h"
#include "m_random.h"
#include "p_map.h"
#include "p_maputl.h"
#include "p_mobj.h"
#include "p_tick.h"
#include "r_main.h"
#include "r_state.h"
#include <emscripten/emscripten.h>

#define API_VERSION 1
#define ACTION_ATTACK 1
#define ACTION_USE 2

static ticcmd_t queued_action;
static int queued_action_enabled;
static char json_buffer[1024];
static int teacher_enabled;
static int teacher_start_gametic;
static int teacher_max_tics;
static int teacher_capture_every;
static int teacher_sample_count;
static int teacher_last_episode;
static int teacher_last_map;
static unsigned int teacher_seed;
static int teacher_search_sign = 1;
static int teacher_death_latched;
static int teacher_death_count;
static fixed_t teacher_last_x;
static fixed_t teacher_last_y;
static int teacher_stagnant_tics;
static int teacher_recovery_until;

extern void D_Display(fixed_t frac);
extern void D_AgentLoopIter(void);
extern dboolean singletics;

static double map_unit(fixed_t value) { return (double)value / FRACUNIT; }
static mobj_t *player_mobj(void) { return players[consoleplayer].mo; }
static int teacher_check_position(mobj_t *mo, fixed_t x, fixed_t y)
{
  int clear = P_CheckPosition(mo, x, y);
  /* P_CheckPosition uses the engine's per-tic scratch target. Teacher probes
     happen during command construction, so they must not leak into P_Ticker. */
  P_MapEnd();
  return clear;
}

EMSCRIPTEN_KEEPALIVE int agent_api_version(void) { return API_VERSION; }

EMSCRIPTEN_KEEPALIVE int agent_reset(int skill, int episode, int map, unsigned int seed)
{
  if (skill < sk_baby || skill > sk_nightmare || episode < 1 || map < 1)
    return 0;
  memset(&queued_action, 0, sizeof queued_action);
  queued_action_enabled = 1;
  rngseed = seed ? seed : 1;
  M_ClearRandom();
  paused = 0;
  menuactive = mnact_inactive;
  G_InitNew((skill_t)skill, episode, map);
  /* G_InitNew schedules the load. One action-free tick realizes it. */
  memset(&netcmds[consoleplayer][(gametic / ticdup) % BACKUPTICS], 0,
         sizeof(ticcmd_t));
  G_Ticker();
  gametic++;
  return player_mobj() != NULL;
}

EMSCRIPTEN_KEEPALIVE int agent_set_action(int forward, int strafe, int turn, int buttons)
{
  paused = 0;
  menuactive = mnact_inactive;
  if (forward < -127) forward = -127; if (forward > 127) forward = 127;
  if (strafe < -127) strafe = -127; if (strafe > 127) strafe = 127;
  if (turn < -32768) turn = -32768; if (turn > 32767) turn = 32767;
  memset(&queued_action, 0, sizeof queued_action);
  queued_action.forwardmove = (signed char)forward;
  queued_action.sidemove = (signed char)strafe;
  queued_action.angleturn = (signed short)turn;
  if (buttons & ACTION_ATTACK) queued_action.buttons |= BT_ATTACK;
  if (buttons & ACTION_USE) queued_action.buttons |= BT_USE;
  queued_action_enabled = 1;
  return 1;
}

void agent_override_ticcmd(ticcmd_t *cmd)
{
  if (teacher_enabled && cmd && player_mobj()) {
    mobj_t *player = player_mobj();
    mobj_t *best = NULL;
    thinker_t *th;
    fixed_t best_dist = INT_MAX;
    int elapsed = gametic - teacher_start_gametic;
    fixed_t progress;

    memset(&queued_action, 0, sizeof queued_action);
    progress = P_AproxDistance(player->x - teacher_last_x, player->y - teacher_last_y);
    if (progress < FRACUNIT / 8)
      teacher_stagnant_tics++;
    else
      teacher_stagnant_tics = 0;
    teacher_last_x = player->x;
    teacher_last_y = player->y;
    if (teacher_stagnant_tics >= 10 && gametic >= teacher_recovery_until) {
      teacher_search_sign = -teacher_search_sign;
      teacher_recovery_until = gametic + 28;
      teacher_stagnant_tics = 0;
    }
    if (gametic < teacher_recovery_until) {
      queued_action.forwardmove = -18;
      queued_action.sidemove = teacher_search_sign * 32;
      queued_action.angleturn = teacher_search_sign * 900;
      if ((teacher_recovery_until - gametic) > 24)
        queued_action.buttons |= BT_USE;
      queued_action_enabled = 1;
      goto copy_teacher_action;
    }
    for (th = thinkercap.next; th != &thinkercap; th = th->next) {
      mobj_t *mo;
      fixed_t dist;
      if (th->function != P_MobjThinker) continue;
      mo = (mobj_t *)th;
      if (!ALIVE(mo) || !P_CheckSight(player, mo)) continue;
      dist = P_AproxDistance(mo->x - player->x, mo->y - player->y);
      if (dist < best_dist) { best = mo; best_dist = dist; }
    }
    if (best) {
      angle_t desired = R_PointToAngle2(player->x, player->y, best->x, best->y);
      signed int delta = (signed int)(desired - player->angle);
      int turn = delta / 65536;
      if (turn < -900) turn = -900;
      if (turn > 900) turn = 900;
      queued_action.angleturn = (signed short)turn;
      queued_action.forwardmove = abs(turn) < 180 ? 28 : 8;
      queued_action.sidemove = abs(turn) < 180 ? ((elapsed / 35) & 1 ? 18 : -18) : 0;
      if (abs(turn) < 115) queued_action.buttons |= BT_ATTACK;
      /* Combat must not suppress interaction forever. If the authoritative
         movement probe says the facing path is blocked, try the nearby door
         while keeping the target-facing command deterministic. */
      if (!teacher_check_position(player,
            player->x + FixedMul(36 * FRACUNIT, finecosine[player->angle >> ANGLETOFINESHIFT]),
            player->y + FixedMul(36 * FRACUNIT, finesine[player->angle >> ANGLETOFINESHIFT])))
        queued_action.buttons |= BT_USE;
    } else if (teacher_check_position(player,
               player->x + FixedMul(40 * FRACUNIT, finecosine[player->angle >> ANGLETOFINESHIFT]),
               player->y + FixedMul(40 * FRACUNIT, finesine[player->angle >> ANGLETOFINESHIFT]))) {
      queued_action.forwardmove = 45;
      if (elapsed % 35 == 0) queued_action.buttons |= BT_USE;
    } else {
      if (elapsed % 24 == 0) teacher_search_sign = -teacher_search_sign;
      queued_action.forwardmove = 10;
      queued_action.sidemove = teacher_search_sign * 20;
      queued_action.angleturn = teacher_search_sign * 720;
      if (elapsed % 18 == 0) queued_action.buttons |= BT_USE;
    }
    queued_action_enabled = 1;
  }
copy_teacher_action:
  if (!queued_action_enabled || !cmd) return;
  cmd->forwardmove = queued_action.forwardmove;
  cmd->sidemove = queued_action.sidemove;
  cmd->angleturn = queued_action.angleturn;
  cmd->buttons = queued_action.buttons;
}

EMSCRIPTEN_KEEPALIVE void agent_teacher_begin(unsigned int seed, int max_tics, int capture_every)
{
  teacher_seed = seed ? seed : 1;
  rngseed = teacher_seed;
  M_ClearRandom();
  paused = 0;
  menuactive = mnact_inactive;
  teacher_start_gametic = gametic;
  teacher_max_tics = max_tics > 0 ? max_tics : 350;
  teacher_capture_every = capture_every > 0 ? capture_every : 5;
  teacher_sample_count = 0;
  teacher_last_episode = gameepisode;
  teacher_last_map = gamemap;
  teacher_search_sign = 1;
  teacher_death_latched = 0;
  teacher_death_count = 0;
  teacher_last_x = player_mobj() ? player_mobj()->x : 0;
  teacher_last_y = player_mobj() ? player_mobj()->y : 0;
  teacher_stagnant_tics = 0;
  teacher_recovery_until = 0;
  teacher_enabled = 1;
  queued_action_enabled = 1;
}

int agent_teacher_active(void) { return teacher_enabled; }

void agent_teacher_after_render(void)
{
  int elapsed;
  const char *state;
  if (!teacher_enabled) return;
  elapsed = gametic - teacher_start_gametic;
  if (elapsed < 0) elapsed = 0;
  if ((elapsed % teacher_capture_every) == 0) {
    state = agent_state_json();
    EM_ASM({
      if (window.__dwasmTeacherSample) {
        window.__dwasmTeacherSample(UTF8ToString($0), $1, $2, $3, $4, $5, $6);
      }
    }, state, queued_action.forwardmove, queued_action.sidemove,
       queued_action.angleturn, !!(queued_action.buttons & BT_ATTACK),
       !!(queued_action.buttons & BT_USE), teacher_sample_count++);
  }
  if (gameepisode != teacher_last_episode || gamemap != teacher_last_map) {
    teacher_last_episode = gameepisode;
    teacher_last_map = gamemap;
    EM_ASM({ if (window.__dwasmTeacherBoundary) window.__dwasmTeacherBoundary('level'); });
  }
  if (players[consoleplayer].playerstate == PST_DEAD && !teacher_death_latched) {
    teacher_death_latched = 1;
    teacher_death_count++;
    EM_ASM({ if (window.__dwasmTeacherBoundary) window.__dwasmTeacherBoundary('death'); });
    /* Schedule a deterministic level restart for G_Ticker. Never initialize
       a level directly from the post-render callback. */
    rngseed = teacher_seed + (unsigned int)teacher_death_count * 2654435761u;
    M_ClearRandom();
    G_DeferedInitNew(gameskill, gameepisode, gamemap);
  } else if (players[consoleplayer].playerstate != PST_DEAD) {
    teacher_death_latched = 0;
  }
  if (elapsed >= teacher_max_tics) {
    teacher_enabled = 0;
    memset(&queued_action, 0, sizeof queued_action);
    EM_ASM({ if (window.__dwasmTeacherComplete) window.__dwasmTeacherComplete(); });
  }
}

EMSCRIPTEN_KEEPALIVE int agent_step(int tics)
{
  int i;
  if (tics < 1) return 0;
  if (tics > 35) tics = 35; /* bounded to one simulated second per call */
  singletics = true;
  for (i = 0; i < tics; ++i) {
    D_AgentLoopIter();
  }
  return tics;
}

EMSCRIPTEN_KEEPALIVE int agent_render(void)
{
  if (!player_mobj()) return 0;
  D_Display(FRACUNIT);
  return 1;
}

EMSCRIPTEN_KEEPALIVE int agent_pause(void)
{
  emscripten_pause_main_loop();
  return 1;
}

EMSCRIPTEN_KEEPALIVE int agent_resume(void)
{
  emscripten_resume_main_loop();
  return 1;
}

EMSCRIPTEN_KEEPALIVE const char *agent_state_json(void)
{
  player_t *p = &players[consoleplayer];
  mobj_t *mo = p->mo;
  if (!mo) { strcpy(json_buffer, "{\"ready\":false}"); return json_buffer; }
  snprintf(json_buffer, sizeof json_buffer,
    "{\"ready\":true,\"tic\":%d,\"level_time\":%d,\"episode\":%d,\"map\":%d,"
    "\"x\":%.4f,\"y\":%.4f,\"z\":%.4f,\"angle\":%u,"
    "\"vx\":%.4f,\"vy\":%.4f,\"health\":%d,\"armor\":%d,"
    "\"weapon\":%d,\"ammo\":[%d,%d,%d,%d],\"kills\":%d,"
    "\"items\":%d,\"secrets\":%d,\"dead\":%s,"
    "\"teacher\":{\"active\":%s,\"seed\":%u,\"start_tic\":%d,"
    "\"max_tics\":%d,\"capture_every\":%d}}",
    gametic, leveltime, gameepisode, gamemap,
    map_unit(mo->x), map_unit(mo->y), map_unit(mo->z), mo->angle,
    map_unit(mo->momx), map_unit(mo->momy), mo->health, p->armorpoints,
    p->readyweapon, p->ammo[0], p->ammo[1], p->ammo[2], p->ammo[3],
    p->killcount, p->itemcount, p->secretcount,
    p->playerstate == PST_DEAD ? "true" : "false",
    teacher_enabled ? "true" : "false", teacher_seed, teacher_start_gametic,
    teacher_max_tics, teacher_capture_every);
  return json_buffer;
}

EMSCRIPTEN_KEEPALIVE int agent_probe_move(int forward_units, int strafe_units)
{
  mobj_t *mo = player_mobj();
  double a, dx, dy;
  if (!mo) return -1;
  a = (double)mo->angle * (6.28318530717958647692 / 4294967296.0);
  dx = cos(a) * forward_units - sin(a) * strafe_units;
  dy = sin(a) * forward_units + cos(a) * strafe_units;
  return teacher_check_position(mo, mo->x + (fixed_t)(dx * FRACUNIT),
                                mo->y + (fixed_t)(dy * FRACUNIT)) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int agent_map_line_count(void) { return numlines; }

EMSCRIPTEN_KEEPALIVE const char *agent_map_line_json(int index)
{
  line_t *line;
  if (index < 0 || index >= numlines) return "{}";
  line = &lines[index];
  snprintf(json_buffer, sizeof json_buffer,
    "{\"index\":%d,\"x1\":%.4f,\"y1\":%.4f,\"x2\":%.4f,\"y2\":%.4f,"
    "\"flags\":%u,\"special\":%d,\"tag\":%d,\"two_sided\":%s}",
    index, map_unit(line->v1->x), map_unit(line->v1->y),
    map_unit(line->v2->x), map_unit(line->v2->y), line->flags,
    line->special, line->tag, line->backsector ? "true" : "false");
  return json_buffer;
}

static mobj_t *enemy_at(int wanted)
{
  thinker_t *th; int found = 0;
  for (th = thinkercap.next; th != &thinkercap; th = th->next) {
    mobj_t *mo;
    if (th->function != P_MobjThinker) continue;
    mo = (mobj_t *)th;
    if (!ALIVE(mo)) continue;
    if (found++ == wanted) return mo;
  }
  return NULL;
}

EMSCRIPTEN_KEEPALIVE int agent_enemy_count(void)
{
  thinker_t *th; int count = 0;
  for (th = thinkercap.next; th != &thinkercap; th = th->next) {
    mobj_t *mo;
    if (th->function != P_MobjThinker) continue;
    mo = (mobj_t *)th;
    if (ALIVE(mo)) count++;
  }
  return count;
}

EMSCRIPTEN_KEEPALIVE const char *agent_enemy_json(int index)
{
  mobj_t *enemy = enemy_at(index); mobj_t *player = player_mobj();
  if (!enemy) return "{}";
  snprintf(json_buffer, sizeof json_buffer,
    "{\"index\":%d,\"type\":%d,\"x\":%.4f,\"y\":%.4f,\"z\":%.4f,"
    "\"angle\":%u,\"health\":%d,\"radius\":%.4f,\"visible\":%s}",
    index, enemy->type, map_unit(enemy->x), map_unit(enemy->y), map_unit(enemy->z),
    enemy->angle, enemy->health, map_unit(enemy->radius),
    player && P_CheckSight(player, enemy) ? "true" : "false");
  return json_buffer;
}
