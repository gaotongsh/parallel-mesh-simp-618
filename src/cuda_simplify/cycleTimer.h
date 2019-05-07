#ifndef CYCLETIMER_H
/* Cycle timer code, adapted from CycleTimer.h found in 15-418 code repositories */

#ifndef TIME
#define TIME 0
#endif

#if TIME
#define WATCH_OPS(x) x
#else
#define WATCH_OPS(x)
#endif

#ifdef __cplusplus
extern "C" {
#endif

double currentSeconds();

void watchStart(char* clockName);

void watchTake(char* sampleName);

void watchReport();

#ifdef __cplusplus
}
#endif

#define CYCLETIMER_H
#endif