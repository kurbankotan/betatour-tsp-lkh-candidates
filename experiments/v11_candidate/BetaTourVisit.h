/* Shared, low-overhead counters for action-edge enumeration in LK loops. */
#ifndef BETATOUR_VISIT_H
#define BETATOUR_VISIT_H

extern int *BT_ActionTo;
extern unsigned long long *BT_ActionVisits;
extern unsigned long long *BT_ActionFirstGatePasses;
extern unsigned long long *BT_ActionPositiveG1;
extern unsigned long long *BT_ActionAcceptedEntries;
extern unsigned long long BT_TotalCandidateEnumerations;
extern unsigned long long BT_TotalActionEnumerations;
extern unsigned long long BT_TotalTopLevelEnumerations;
extern unsigned long long BT_TotalTopLevelActionEnumerations;
extern unsigned long long BT_TotalActionFirstGatePasses;
extern unsigned long long BT_TotalActionPositiveG1;
extern unsigned long long BT_TotalActionAcceptedEntries;
void BT_FlushTelemetryPeriodic(void);

/* Counts a directed candidate when a patched search loop enumerates it.
 * Eligibility and gain filters may still reject it later. */
static inline void BT_NoteCandidate(Node *From, Node *To)
{
    if (!BT_ActionTo || !To)
        return;
    BT_TotalCandidateEnumerations++;
    if (BT_ActionTo[From->Id] == To->Id) {
        BT_TotalActionEnumerations++;
        BT_ActionVisits[From->Id]++;
    }
    if ((BT_TotalCandidateEnumerations & 65535ULL) == 0)
        BT_FlushTelemetryPeriodic();
}

/* This is the first t2->t3 choice in Best2/3/4/5/K/SpecialOptMove only. */
static inline void BT_NoteTopLevelEnumerated(Node *From, Node *To)
{
    if (!BT_ActionTo)
        return;
    BT_TotalTopLevelEnumerations++;
    if (BT_ActionTo[From->Id] == To->Id)
        BT_TotalTopLevelActionEnumerations++;
}

/* Called after that loop's first adjacency/G1 gate. This is not the full
 * multi-step gain filter, and G1 > 0 is tracked separately. */
static inline void BT_NoteTopLevelPass(Node *From, Node *To, GainType G1)
{
    if (!BT_ActionTo || BT_ActionTo[From->Id] != To->Id)
        return;
    BT_ActionFirstGatePasses[From->Id]++;
    BT_TotalActionFirstGatePasses++;
    if (G1 > 0) {
        BT_ActionPositiveG1[From->Id]++;
        BT_TotalActionPositiveG1++;
    }
}

/* StoreTour calls this before replacing OldPred/OldSuc on affected nodes.
 * It counts an action edge newly present in a locally accepted LK tour. */
static inline void BT_NoteAcceptedEdge(Node *From)
{
    int To;
    if (!BT_ActionTo || !(To = BT_ActionTo[From->Id]))
        return;
    if ((From->Pred->Id == To || From->Suc->Id == To) &&
        From->OldPred->Id != To && From->OldSuc->Id != To) {
        BT_ActionAcceptedEntries[From->Id]++;
        BT_TotalActionAcceptedEntries++;
    }
}

#endif
