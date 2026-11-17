/* Mock-LKH check of v15 barrier swaps, protected edges and telemetry. */
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mock_lkh.h"
#include "../BetaTourVisit.h"
#include "../BetaTourSwapHook.h"

#define N 12

static int Wrap(int Id)
{
    while (Id < 1) Id += N;
    while (Id > N) Id -= N;
    return Id;
}

static int RowLength(int From)
{
    int i = 0;
    while (NodeSet[From].CandidateSet[i].To) i++;
    return i;
}

static int RowIndex(int From, int To)
{
    Candidate *Row = NodeSet[From].CandidateSet;
    int i;
    for (i = 0; Row[i].To; i++)
        if (Row[i].To->Id == To) return i;
    return -1;
}

static void SetTour(const int *Order)
{
    int i;
    for (i = 1; i <= N; i++) BetterTour[i] = Order[i - 1];
    BetterTour[0] = BetterTour[N];
}

int main(int argc, char **argv)
{
    int i, k, Identity[N], Noop, Omit, Reject;
    FILE *File;
    char Path[PATH_MAX];
    const int Offsets[6] = {1, -1, 2, -2, 3, -3};
    const int Alphas[6] = {0, 0, 10, 20, 30, 40};
    int Better[N] = {1, 7, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12};

    assert(argc == 2 || argc == 3);
    Noop = argc == 3 && strcmp(argv[2], "noop") == 0;
    Omit = argc == 3 && strcmp(argv[2], "omit") == 0;
    Reject = argc == 3 && strcmp(argv[2], "reject") == 0;
    setenv("BETATOUR_HOOK_DIR", argv[1], 1);
    setenv("BETATOUR_HOOK_TIMEOUT_S", "5", 1);
    Dimension = DimensionSaved = N;
    NodeSet = calloc(N + 1, sizeof(Node));
    BetterTour = calloc(N + 1, sizeof(int));
    for (i = 1; i <= N; i++) NodeSet[i].Id = i;
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
    BT_ExportNativeCandidates();
    Trial = 1;
    BT_RecordBest();

    /* All six ensemble tours can be identical in this unit fixture. */
    snprintf(Path, sizeof(Path), "%s/protected.edges", argv[1]);
    File = fopen(Path, "w");
    assert(File);
    fprintf(File, "%d %d\n", N, N);
    for (i = 1; i < N; i++) fprintf(File, "%d %d\n", i, i + 1);
    if (Omit) fprintf(File, "2 %d\n", N);
    else fprintf(File, "1 %d\n", N);
    fclose(File);

    snprintf(Path, sizeof(Path), "%s/action.txt", argv[1]);
    File = fopen(Path, "w");
    assert(File);
    fprintf(File, "%d\n", N);
    for (i = 1; i <= N; i++) {
        if (Reject && i == 1) fprintf(File, "1 8 2 500\n");
        else if (!Noop && i == 1) fprintf(File, "1 7 3 500\n");
        else if (!Noop && i == 5) fprintf(File, "5 11 7 500\n");
        else if (!Noop && i == 9) fprintf(File, "9 3 11 500\n");
        else fprintf(File, "%d 0 0 0\n", i);
    }
    fclose(File);
    BT_AfterFirstTrial();
    if (Noop) {
        for (i = 1; i <= N; i++) assert(RowLength(i) == 6);
        assert(!BT_ActionTo[1] && !BT_ActionTo[5] && !BT_ActionTo[9]);
        puts("v15 hook no-op test passed");
        return 0;
    }
    assert(BT_ProtectedCount == N && BT_IsProtected(1, 2));
    assert(RowIndex(1, 7) == 2 && RowIndex(5, 11) == 2 && RowIndex(9, 3) == 2);
    assert(RowIndex(1, 3) < 0 && RowIndex(5, 7) < 0 && RowIndex(9, 11) < 0);
    for (i = 1; i <= N; i++) assert(RowLength(i) == 6);
    assert(BT_ActionVictim[1] == 3 && BT_ActionVictim[5] == 7);

    /* Protected edge, top-two victim, reverse duplicate, and missing victim. */
    assert(BT_ApplyOne(1, 8, 2, 500) == -1);
    assert(strcmp(BT_LastSwapRejectReason,
                  "victim is a protected first-tour edge") == 0);
    {
        size_t ProtectedCount = BT_ProtectedCount;
        BT_ProtectedCount = 0;
        assert(BT_ApplyOne(1, 8, 12, 500) == -1);
        assert(strcmp(BT_LastSwapRejectReason,
                      "victim is in a protected top-two slot") == 0);
        BT_ProtectedCount = ProtectedCount;
    }
    assert(BT_ApplyOne(1, 7, 4, 500) == -1);
    assert(strcmp(BT_LastSwapRejectReason,
                  "target already exists in source row") == 0);
    assert(BT_ApplyOne(1, 3, 4, 500) == -1);
    assert(strcmp(BT_LastSwapRejectReason,
                  "target already exists in reverse row") == 0);
    assert(BT_ApplyOne(1, 8, 9, 500) == -1);
    assert(strcmp(BT_LastSwapRejectReason,
                  "victim is missing from source row") == 0);
    assert(RowIndex(1, 7) == 2 && RowLength(1) == 6);

    BT_NoteCandidate(&NodeSet[1], &NodeSet[7]);
    BT_NoteTopLevelEnumerated(&NodeSet[1], &NodeSet[7]);
    BT_NoteTopLevelPass(&NodeSet[1], &NodeSet[7], 5);
    NodeSet[1].Suc = &NodeSet[7];
    BT_NoteAcceptedEdge(&NodeSet[1]);
    SetTour(Better);
    BetterCost = 990;
    BT_RecordBest();
    Trial = 2;
    BT_AfterFirstTrial();
    assert(BT_TotalActionEnumerations == 1 && BT_ActionIncumbentEntries == 1);
    puts("v15 hook logic test passed");
    return 0;
}
