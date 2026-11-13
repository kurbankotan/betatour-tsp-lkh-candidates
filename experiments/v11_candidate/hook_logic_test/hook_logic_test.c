/* Unit test of the BetaTour v11 hook logic against a mock LKH state.
 * Build and run through test_hook_logic.py; it checks the files written here.
 * Scenario: 12 nodes, three actions at slot 3, then two kinds of LKH-style
 * row reordering between trials, then a better tour containing one action. */
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mock_lkh.h"
#include "BetaTourVisit.h"
#include "../BetaTourSearchHook.h"

#define N 12

static int Wrap(int Id)
{
    while (Id < 1)
        Id += N;
    while (Id > N)
        Id -= N;
    return Id;
}

static int RowIndex(int From, int To)
{
    Candidate *Start = NodeSet[From].CandidateSet;
    int i;
    for (i = 0; Start[i].To; i++)
        if (Start[i].To->Id == To)
            return i;
    return -1;
}

static int RowLength(int From)
{
    int i = 0;
    while (NodeSet[From].CandidateSet[i].To)
        i++;
    return i;
}

/* LKH-style reorder variant A: insertion sort on (Alpha, Cost). */
static void SortRowByAlpha(int From)
{
    Candidate *Start = NodeSet[From].CandidateSet, Temp, *NFrom, *NN;
    for (NFrom = Start + 1; NFrom->To; NFrom++) {
        Temp = *NFrom;
        for (NN = NFrom - 1; NN >= Start &&
             (Temp.Alpha < NN->Alpha ||
              (Temp.Alpha == NN->Alpha && Temp.Cost < NN->Cost)); NN--)
            *(NN + 1) = *NN;
        *(NN + 1) = Temp;
    }
}

/* LKH-style reorder variant B: move one entry to the start of the row. */
static void MoveToFront(int From, int Index)
{
    Candidate *Start = NodeSet[From].CandidateSet, Temp = Start[Index];
    memmove(Start + 1, Start, (size_t) Index * sizeof(Candidate));
    Start[0] = Temp;
}

static void SetTour(const int *Order)
{
    int i;
    for (i = 1; i <= N; i++)
        BetterTour[i] = Order[i - 1];
    BetterTour[0] = BetterTour[N];
}

int main(int argc, char **argv)
{
    int i, k;
    FILE *Action;
    char Path[PATH_MAX];
    int Identity[N], Better[N] = {1, 7, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12};
    const int Offsets[6] = {1, -1, 2, -2, 3, -3};
    const int Alphas[6] = {0, 0, 10, 20, 30, 40};
    int Before[N + 1][16], BeforeLength;

    assert(argc == 2);
    setenv("BETATOUR_HOOK_DIR", argv[1], 1);
    setenv("BETATOUR_HOOK_TIMEOUT_S", "5", 1);
    Dimension = DimensionSaved = N;
    NodeSet = calloc(N + 1, sizeof(Node));
    BetterTour = calloc(N + 1, sizeof(int));
    for (i = 1; i <= N; i++)
        NodeSet[i].Id = i;
    for (i = 1; i <= N; i++) {
        NodeSet[i].Suc = &NodeSet[Wrap(i + 1)];
        NodeSet[i].Pred = &NodeSet[Wrap(i - 1)];
        NodeSet[i].OldSuc = NodeSet[i].Suc;
        NodeSet[i].OldPred = NodeSet[i].Pred;
        for (k = 0; k < 6; k++)
            assert(AddCandidate(&NodeSet[i], &NodeSet[Wrap(i + Offsets[k])],
                                C(&NodeSet[i], &NodeSet[Wrap(i + Offsets[k])]),
                                Alphas[k]));
        Identity[i - 1] = i;
    }
    SetTour(Identity);
    BetterCost = 1000;

    /* Trial 1: native export, first better tour, barrier and actions. */
    BT_ExportNativeCandidates();
    Trial = 1;
    BT_RecordBest();
    snprintf(Path, sizeof(Path), "%s/action.txt", argv[1]);
    Action = fopen(Path, "w");
    assert(Action);
    fprintf(Action, "%d\n", N);
    for (i = 1; i <= N; i++) {
        if (i == 1)
            fprintf(Action, "1 7 3 500\n");
        else if (i == 5)
            fprintf(Action, "5 11 3 500\n");
        else if (i == 9)
            fprintf(Action, "9 3 3 500\n");
        else
            fprintf(Action, "%d 0 0 500\n", i);
    }
    fclose(Action);
    BT_AfterFirstTrial();
    assert(RowIndex(1, 7) == 2 && RowIndex(5, 11) == 2 && RowIndex(9, 3) == 2);
    assert(RowLength(1) == 7 && RowLength(2) == 6);
    assert(BT_ActionSlot[1] == 3 && BT_ActionSlot[2] == 0);

    /* Trial 2: variant A demotes every action to the end of its row. */
    SortRowByAlpha(1);
    SortRowByAlpha(5);
    SortRowByAlpha(9);
    assert(RowIndex(1, 7) == 6);
    Trial = 2;
    BT_AfterFirstTrial();
    assert(RowIndex(1, 7) == 2 && RowIndex(5, 11) == 2 && RowIndex(9, 3) == 2);
    assert(BT_SlotRestorations == 3 && BT_SlotRestorationDisplacement == 12);

    /* Trial 3: variant B moves two native entries of node 1 in front of the
     * action and moves node 5's action itself to the front. */
    BeforeLength = RowLength(1);
    for (k = 0; k < BeforeLength; k++)
        Before[1][k] = NodeSet[1].CandidateSet[k].To->Id;
    MoveToFront(1, 4);
    MoveToFront(1, 5);
    assert(RowIndex(1, 7) == 4);
    MoveToFront(5, 2);
    assert(RowIndex(5, 11) == 0);
    BT_LastFullTelemetryWall = 0; /* force a full TSV snapshot */
    Trial = 3;
    BT_AfterFirstTrial();
    assert(RowIndex(1, 7) == 2 && RowIndex(5, 11) == 0 && RowIndex(9, 3) == 2);
    /* Node 1 must read: moved[5], moved[4], action, then the rest in order. */
    assert(NodeSet[1].CandidateSet[0].To->Id == Before[1][5]);
    assert(NodeSet[1].CandidateSet[1].To->Id == Before[1][4]);
    assert(NodeSet[1].CandidateSet[3].To->Id == Before[1][0]);
    assert(NodeSet[1].CandidateSet[4].To->Id == Before[1][1]);
    assert(NodeSet[1].CandidateSet[5].To->Id == Before[1][3]);
    assert(NodeSet[1].CandidateSet[6].To->Id == Before[1][6]);
    assert(NodeSet[1].CandidateSet[2].Alpha == 500);
    assert(BT_SlotRestorations == 4 && BT_SlotRestorationDisplacement == 14);
    assert(BT_SlotFoundAhead == 1 && BT_SlotAheadAtLastCheck == 1);
    assert(BT_SlotMissing == 0 && BT_SlotChecks == 2);
    assert(BT_ActionSlotRestorations[1] == 2 &&
           BT_ActionSlotRestorations[5] == 1 &&
           BT_ActionSlotRestorations[9] == 1);

    /* Search counters: two enumerations of action 1->7, one of a non-action
     * edge, a top-level gate pass and an accepted LK move that adds 1-7. */
    BT_NoteCandidate(&NodeSet[1], &NodeSet[7]);
    BT_NoteCandidate(&NodeSet[1], &NodeSet[7]);
    BT_NoteCandidate(&NodeSet[1], &NodeSet[2]);
    BT_NoteTopLevelEnumerated(&NodeSet[1], &NodeSet[7]);
    BT_NoteTopLevelPass(&NodeSet[1], &NodeSet[7], 5);
    NodeSet[1].Suc = &NodeSet[7];
    BT_NoteAcceptedEdge(&NodeSet[1]);
    BT_NoteAcceptedEdge(&NodeSet[2]);
    assert(BT_TotalCandidateEnumerations == 3 &&
           BT_TotalActionEnumerations == 2 && BT_ActionVisits[1] == 2);
    assert(BT_TotalActionFirstGatePasses == 1 &&
           BT_TotalActionPositiveG1 == 1 &&
           BT_TotalActionAcceptedEntries == 1);

    /* A better tour after the barrier contains action 1->7. */
    SetTour(Better);
    BetterCost = 990;
    BT_RecordBest();
    assert(BT_ActionIncumbentEntries == 1 && BT_IncumbentEntries[1] == 1);
    assert(BT_LastBetterPresent[1] == 1 && BT_LastBetterPresent[5] == 0);
    BT_LastFullTelemetryWall = 0;
    Trial = 4;
    BT_AfterFirstTrial();
    puts("hook logic test passed");
    return 0;
}
