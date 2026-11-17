/* BetaTour v15 in-place candidate swap hook for LKH 3.0.14. */
#ifndef BETATOUR_HOOK_H
#define BETATOUR_HOOK_H

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

/* One process handles one seed. These globals are read by the patched LK
 * candidate loops through BetaTourVisit.h. */
int *BT_ActionTo = 0;
static int *BT_ActionVictim = 0;
unsigned long long *BT_ActionVisits = 0;
unsigned long long *BT_ActionFirstGatePasses = 0;
unsigned long long *BT_ActionPositiveG1 = 0;
unsigned long long *BT_ActionAcceptedEntries = 0;
unsigned long long BT_TotalCandidateEnumerations = 0;
unsigned long long BT_TotalActionEnumerations = 0;
unsigned long long BT_TotalTopLevelEnumerations = 0;
unsigned long long BT_TotalTopLevelActionEnumerations = 0;
unsigned long long BT_TotalActionFirstGatePasses = 0;
unsigned long long BT_TotalActionPositiveG1 = 0;
unsigned long long BT_TotalActionAcceptedEntries = 0;
static unsigned long long *BT_IncumbentEntries = 0;
static unsigned char *BT_LastBetterPresent = 0;
static unsigned char *BT_EverBetterPresent = 0;
static int *BT_BetterNext = 0, *BT_BetterPrev = 0;
static unsigned long long BT_BetterTourEventsAfterAction = 0;
static unsigned long long BT_ActionIncumbentEntries = 0;
static unsigned long long BT_TrialsCompletedAfterAction = 0;
static int BT_ActionTsvSnapshotTrial = 0;
static double BT_LastFullTelemetryWall = 0;

/* These are all six first-trial tour edges, supplied before action.txt.
 * The hook checks the local seed's tour is covered and refuses to evict an
 * edge from this set. Keys are canonical undirected node-ID pairs. */
static unsigned long long *BT_Protected = 0;
static size_t BT_ProtectedCount = 0;

static const char *BT_HookDirectory(void)
{
    const char *Path = getenv("BETATOUR_HOOK_DIR");
    return Run == 1 && ProblemType == TSP && Dimension == DimensionSaved &&
        Path && *Path ? Path : 0;
}

static void BT_Path(char *Out, size_t Size, const char *Directory,
                    const char *Name)
{
    if (snprintf(Out, Size, "%s/%s", Directory, Name) >= (int) Size) {
        fprintf(stderr, "BetaTour hook path exceeds PATH_MAX\n");
        exit(97);
    }
}

static double BT_WallTime(void)
{
    struct timeval Tv;
    gettimeofday(&Tv, 0);
    return Tv.tv_sec + Tv.tv_usec / 1000000.0;
}

static FILE *BT_OpenTemporary(const char *Directory, const char *Name,
                              char *Final, char *Temporary)
{
    FILE *File;
    BT_Path(Final, PATH_MAX, Directory, Name);
    if (snprintf(Temporary, PATH_MAX, "%s.tmp.%ld", Final, (long) getpid())
        >= PATH_MAX) {
        fprintf(stderr, "BetaTour hook temporary path exceeds PATH_MAX\n");
        exit(97);
    }
    File = fopen(Temporary, "w");
    if (!File) {
        fprintf(stderr, "BetaTour hook cannot write %s: %s\n", Temporary,
                strerror(errno));
        exit(97);
    }
    return File;
}

static void BT_Commit(FILE *File, const char *Temporary, const char *Final)
{
    if (fclose(File) != 0 || rename(Temporary, Final) != 0) {
        fprintf(stderr, "BetaTour hook cannot commit %s: %s\n", Final,
                strerror(errno));
        exit(97);
    }
}

static unsigned long long BT_EdgeKey(int U, int V)
{
    unsigned int A = (unsigned int) (U < V ? U : V);
    unsigned int B = (unsigned int) (U < V ? V : U);
    return ((unsigned long long) A << 32) | B;
}

static int BT_CompareKeys(const void *A, const void *B)
{
    unsigned long long X = *(const unsigned long long *) A;
    unsigned long long Y = *(const unsigned long long *) B;
    return (X > Y) - (X < Y);
}

static int BT_IsProtected(int U, int V)
{
    unsigned long long Key = BT_EdgeKey(U, V);
    return BT_Protected && bsearch(&Key, BT_Protected, BT_ProtectedCount,
                                   sizeof(*BT_Protected), BT_CompareKeys) != 0;
}

/* action.txt is the readiness marker. protected.edges must already exist.
 * Format: N M, followed by M sorted/canonical unique undirected pairs. */
static void BT_ReadProtectedEdges(const char *Directory)
{
    char Path[PATH_MAX], Extra;
    FILE *File;
    int N, M, i, U, V;
    BT_Path(Path, sizeof(Path), Directory, "protected.edges");
    File = fopen(Path, "r");
    if (!File) {
        fprintf(stderr, "BetaTour missing protected edges: %s\n", Path);
        exit(97);
    }
    if (fscanf(File, "%d %d", &N, &M) != 2 || N != DimensionSaved ||
        M < N || M > 6 * N) {
        fprintf(stderr, "BetaTour protected edge header is invalid\n");
        exit(97);
    }
    BT_Protected = calloc((size_t) M, sizeof(*BT_Protected));
    if (!BT_Protected) {
        fprintf(stderr, "BetaTour protected edge allocation failed\n");
        exit(97);
    }
    BT_ProtectedCount = (size_t) M;
    for (i = 0; i < M; i++) {
        if (fscanf(File, "%d %d", &U, &V) != 2 || U < 1 || V > N ||
            U >= V) {
            fprintf(stderr, "BetaTour protected edge row %d is invalid\n", i);
            exit(97);
        }
        BT_Protected[i] = BT_EdgeKey(U, V);
    }
    if (fscanf(File, " %c", &Extra) != EOF) {
        fprintf(stderr, "BetaTour protected edge file has extra data\n");
        exit(97);
    }
    fclose(File);
    qsort(BT_Protected, BT_ProtectedCount, sizeof(*BT_Protected), BT_CompareKeys);
    for (i = 1; i < M; i++)
        if (BT_Protected[i] == BT_Protected[i - 1]) {
            fprintf(stderr, "BetaTour protected edges contain duplicates\n");
            exit(97);
        }
    for (i = 1; i <= N; i++) {
        int A = BetterTour[i], B = BetterTour[i == N ? 1 : i + 1];
        if (A < 1 || A > N || B < 1 || B > N || A == B ||
            !BT_IsProtected(A, B)) {
            fprintf(stderr, "BetaTour protected edges omit seed tour edge\n");
            exit(97);
        }
    }
}

/* Each row: node ID, count, then (neighbor ID, alpha) pairs. */
static void BT_WriteCandidates(const char *Directory, const char *Name)
{
    char Final[PATH_MAX], Temporary[PATH_MAX];
    FILE *File = BT_OpenTemporary(Directory, Name, Final, Temporary);
    int i;
    fprintf(File, "%d\n", DimensionSaved);
    for (i = 1; i <= DimensionSaved; i++) {
        Candidate *CandidatePtr;
        int Count = 0;
        Node *From = &NodeSet[i];
        for (CandidatePtr = From->CandidateSet;
             CandidatePtr && CandidatePtr->To; CandidatePtr++)
            Count++;
        fprintf(File, "%d %d", i, Count);
        for (CandidatePtr = From->CandidateSet;
             CandidatePtr && CandidatePtr->To; CandidatePtr++)
            fprintf(File, " %d %d", CandidatePtr->To->Id,
                    CandidatePtr->Alpha);
        fputc('\n', File);
    }
    BT_Commit(File, Temporary, Final);
}

static void BT_WriteFirstTour(const char *Directory)
{
    char Final[PATH_MAX], Temporary[PATH_MAX];
    FILE *File = BT_OpenTemporary(Directory, "first.tour", Final, Temporary);
    int i;
    fprintf(File, "%d\n", DimensionSaved);
    for (i = 1; i <= DimensionSaved; i++)
        fprintf(File, "%d\n", BetterTour[i]);
    BT_Commit(File, Temporary, Final);
}

/* An incumbent entry is observed only when RecordBetterTour stores a new
 * globally better tour. It is not a count of all accepted LK moves. */
static void BT_ObserveBetterTour(int CountEvent)
{
    int i;
    if (!BT_ActionTo)
        return;
    for (i = 1; i <= DimensionSaved; i++) {
        int From = BetterTour[i];
        int To = BetterTour[i == DimensionSaved ? 1 : i + 1];
        BT_BetterNext[From] = To;
        BT_BetterPrev[To] = From;
    }
    for (i = 1; i <= DimensionSaved; i++) {
        int To = BT_ActionTo[i];
        int Present;
        if (!To)
            continue;
        Present = BT_BetterNext[i] == To || BT_BetterPrev[i] == To;
        if (CountEvent && Present && !BT_LastBetterPresent[i]) {
            BT_IncumbentEntries[i]++;
            BT_ActionIncumbentEntries++;
        }
        BT_LastBetterPresent[i] = Present;
        if (Present)
            BT_EverBetterPresent[i] = 1;
    }
    if (CountEvent)
        BT_BetterTourEventsAfterAction++;
}

static void BT_WriteTelemetry(const char *Directory, int Full)
{
    char Final[PATH_MAX], Temporary[PATH_MAX];
    FILE *File;
    int i, Actions = 0, DistinctVisited = 0, DistinctAccepted = 0;
    int DistinctEver = 0, Current = 0;
    if (!BT_ActionTo)
        return;
    if (Full)
        BT_ActionTsvSnapshotTrial = Trial;
    for (i = 1; i <= DimensionSaved; i++) {
        if (!BT_ActionTo[i])
            continue;
        Actions++;
        DistinctVisited += BT_ActionVisits[i] > 0;
        DistinctAccepted += BT_ActionAcceptedEntries[i] > 0;
        DistinctEver += BT_EverBetterPresent[i] != 0;
        Current += BT_LastBetterPresent[i] != 0;
    }
    File = BT_OpenTemporary(Directory, "telemetry.txt", Final, Temporary);
    fprintf(File, "telemetry_snapshot_epoch_s %.6f\n", BT_WallTime());
    fprintf(File, "action_count %d\n", Actions);
    fprintf(File, "total_candidate_enumerations %llu\n",
            BT_TotalCandidateEnumerations);
    fprintf(File, "action_candidate_enumerations %llu\n",
            BT_TotalActionEnumerations);
    fprintf(File, "distinct_actions_enumerated %d\n", DistinctVisited);
    fprintf(File, "top_level_candidate_enumerations %llu\n",
            BT_TotalTopLevelEnumerations);
    fprintf(File, "top_level_action_enumerations %llu\n",
            BT_TotalTopLevelActionEnumerations);
    fprintf(File, "top_level_action_first_gate_passes %llu\n",
            BT_TotalActionFirstGatePasses);
    fprintf(File, "top_level_action_positive_g1 %llu\n",
            BT_TotalActionPositiveG1);
    fprintf(File, "action_accepted_edge_entries %llu\n",
            BT_TotalActionAcceptedEntries);
    fprintf(File, "distinct_actions_entering_accepted_tours %d\n",
            DistinctAccepted);
    fprintf(File, "better_tour_events_after_action %llu\n",
            BT_BetterTourEventsAfterAction);
    fprintf(File, "action_incumbent_entries %llu\n",
            BT_ActionIncumbentEntries);
    fprintf(File, "distinct_actions_ever_in_better_tour %d\n", DistinctEver);
    fprintf(File, "current_better_tour_actions %d\n", Current);
    fprintf(File, "trials_completed_after_action %llu\n",
            BT_TrialsCompletedAfterAction);
    fprintf(File, "action_tsv_snapshot_trial %d\n", BT_ActionTsvSnapshotTrial);
    fprintf(File, "swap_at_barrier 1\n");
    fprintf(File, "slot_floor_enforced 0\n");
    BT_Commit(File, Temporary, Final);

    /* Per-action rows are refreshed at most every five seconds, so a guided
     * arm with many actions does not pay a per-trial file cost that the sham
     * arm avoids. The small aggregate file above is written for every arm at
     * every completed trial and better tour. */
    if (!Full)
        return;

    File = BT_OpenTemporary(Directory, "action_telemetry.tsv", Final,
                            Temporary);
    fprintf(File, "from\tto\tvictim\tenumerations\tfirst_gate_passes\t"
                  "positive_g1\taccepted_edge_entries\t"
                  "incumbent_entries\t"
                  "ever_in_better_tour\tcurrently_in_better_tour\n");
    for (i = 1; i <= DimensionSaved; i++)
        if (BT_ActionTo[i])
            fprintf(File, "%d\t%d\t%d\t%llu\t%llu\t%llu\t%llu\t%llu\t%d\t%d\n",
                    i, BT_ActionTo[i], BT_ActionVictim[i], BT_ActionVisits[i],
                    BT_ActionFirstGatePasses[i], BT_ActionPositiveG1[i],
                    BT_ActionAcceptedEntries[i],
                    BT_IncumbentEntries[i], BT_EverBetterPresent[i],
                    BT_LastBetterPresent[i]);
    BT_Commit(File, Temporary, Final);
    BT_LastFullTelemetryWall = BT_WallTime();
}

/* Flush live counters during long trials. The callback is reached from the
 * instrumented candidate loops once per 65,536 enumerations, and full
 * snapshots are at least five seconds apart. The final interval can lag;
 * inspect telemetry_snapshot_epoch_s when assessing a deadline run. */
void BT_FlushTelemetryPeriodic(void)
{
    const char *Directory = BT_HookDirectory();
    if (!Directory || !BT_ActionTo)
        return;
    if (BT_WallTime() - BT_LastFullTelemetryWall < 5.0)
        return;
    BT_WriteTelemetry(Directory, 1);
}

/* An atomic incumbent survives an external wall-clock deadline/termination. */
static void BT_RecordBest(void)
{
    const char *Directory = BT_HookDirectory();
    char Final[PATH_MAX], Temporary[PATH_MAX], Events[PATH_MAX];
    FILE *File;
    int i;
    if (!Directory)
        return;
    File = BT_OpenTemporary(Directory, "best.tour", Final, Temporary);
    fprintf(File, "%d\n", DimensionSaved);
    for (i = 1; i <= DimensionSaved; i++)
        fprintf(File, "%d\n", BetterTour[i]);
    BT_Commit(File, Temporary, Final);
    BT_Path(Events, sizeof(Events), Directory, "best.events");
    File = fopen(Events, "a");
    if (!File) {
        fprintf(stderr, "BetaTour hook cannot append %s: %s\n", Events,
                strerror(errno));
        exit(97);
    }
    fprintf(File, "%.6f %lld\n", BT_WallTime(), (long long) BetterCost);
    if (fclose(File) != 0) {
        fprintf(stderr, "BetaTour hook cannot flush %s\n", Events);
        exit(97);
    }
    if (BT_ActionTo) {
        BT_ObserveBetterTour(1);
        BT_WriteTelemetry(Directory, 0);
    }
}

/* LKH runs one search thread per process, so this diagnosis is local to the
 * process and is read immediately after a rejected action. */
static const char *BT_LastSwapRejectReason = "none";

static int BT_ApplyOne(int FromId, int ToId, int VictimId, int Alpha)
{
    Node *From, *To;
    Candidate *Start, *Reverse;
    int Count = 0, VictimIndex = -1, i;

    BT_LastSwapRejectReason = "none";
    if (FromId < 1 || FromId > DimensionSaved || ToId < 1 ||
        ToId > DimensionSaved || ToId == FromId || VictimId < 1 ||
        VictimId > DimensionSaved || VictimId == FromId ||
        VictimId == ToId ||
        Alpha < 0 || Alpha == INT_MAX) {
        BT_LastSwapRejectReason = "invalid node ID or alpha";
        return -1;
    }
    if (BT_IsProtected(FromId, VictimId)) {
        BT_LastSwapRejectReason = "victim is a protected first-tour edge";
        return -1;
    }
    From = &NodeSet[FromId];
    To = &NodeSet[ToId];
    Start = From->CandidateSet;
    if (!Start) {
        BT_LastSwapRejectReason = "source candidate row is missing";
        return -1;
    }
    for (i = 0; Start[i].To; i++) {
        if (Start[i].To == To) {
            BT_LastSwapRejectReason = "target already exists in source row";
            return -1;
        }
        if (Start[i].To->Id == VictimId)
            VictimIndex = i;
        Count++;
    }
    /* The two highest-priority native entries are protected. */
    if (VictimIndex < 0) {
        BT_LastSwapRejectReason = "victim is missing from source row";
        return -1;
    }
    if (VictimIndex < 2) {
        BT_LastSwapRejectReason = "victim is in a protected top-two slot";
        return -1;
    }
    /* The new undirected edge must be absent from the full live graph.
     * A row belongs to one source, so only this reverse row can contain it. */
    Reverse = To->CandidateSet;
    for (i = 0; Reverse && Reverse[i].To; i++)
        if (Reverse[i].To == From) {
            BT_LastSwapRejectReason = "target already exists in reverse row";
            return -1;
        }
    Start[VictimIndex].To = To;
    Start[VictimIndex].Cost = C(From, To);
    Start[VictimIndex].Alpha = Alpha;
    /* In-place replacement must not alter the directed row's length. */
    for (i = 0; Start[i].To; i++)
        ;
    if (i != Count) {
        BT_LastSwapRejectReason = "source row length changed during swap";
        return -1;
    }
    return 0;
}

static void BT_ValidateNoop(int From, int To, int Victim)
{
    if (To == 0 && Victim != 0) {
        fprintf(stderr, "BetaTour malformed no-op at source %d\n", From);
        exit(97);
    }
}

static void BT_WaitAndApply(const char *Directory)
{
    char Action[PATH_MAX], Final[PATH_MAX], Temporary[PATH_MAX];
    FILE *File;
    int N, i, From, To, Victim, Alpha;
    int Swapped = 0, Rejected = 0;
    double Start = BT_WallTime(), Timeout = 300.0;
    const char *TimeoutText = getenv("BETATOUR_HOOK_TIMEOUT_S");
    if (TimeoutText && atof(TimeoutText) > 0)
        Timeout = atof(TimeoutText);
    BT_Path(Action, sizeof(Action), Directory, "action.txt");
    while (!(File = fopen(Action, "r"))) {
        if (BT_WallTime() - Start > Timeout) {
            fprintf(stderr, "BetaTour hook timed out waiting for %s\n", Action);
            exit(97);
        }
        usleep(10000);
    }
    if (fscanf(File, "%d", &N) != 1 || N != DimensionSaved) {
        fprintf(stderr, "BetaTour hook action dimension mismatch\n");
        exit(97);
    }
    BT_ReadProtectedEdges(Directory);
    BT_ActionTo = calloc((size_t) N + 1, sizeof(*BT_ActionTo));
    BT_ActionVictim = calloc((size_t) N + 1, sizeof(*BT_ActionVictim));
    BT_ActionVisits = calloc((size_t) N + 1, sizeof(*BT_ActionVisits));
    BT_ActionFirstGatePasses = calloc((size_t) N + 1,
                                      sizeof(*BT_ActionFirstGatePasses));
    BT_ActionPositiveG1 = calloc((size_t) N + 1,
                                 sizeof(*BT_ActionPositiveG1));
    BT_ActionAcceptedEntries = calloc((size_t) N + 1,
                                      sizeof(*BT_ActionAcceptedEntries));
    BT_IncumbentEntries = calloc((size_t) N + 1,
                                 sizeof(*BT_IncumbentEntries));
    BT_LastBetterPresent = calloc((size_t) N + 1,
                                  sizeof(*BT_LastBetterPresent));
    BT_EverBetterPresent = calloc((size_t) N + 1,
                                  sizeof(*BT_EverBetterPresent));
    BT_BetterNext = calloc((size_t) N + 1, sizeof(*BT_BetterNext));
    BT_BetterPrev = calloc((size_t) N + 1, sizeof(*BT_BetterPrev));
    if (!BT_ActionTo || !BT_ActionVictim || !BT_ActionVisits || !BT_ActionFirstGatePasses ||
        !BT_ActionPositiveG1 || !BT_ActionAcceptedEntries ||
        !BT_IncumbentEntries || !BT_LastBetterPresent ||
        !BT_EverBetterPresent || !BT_BetterNext || !BT_BetterPrev) {
        fprintf(stderr, "BetaTour telemetry allocation failed\n");
        exit(97);
    }
    for (i = 1; i <= N; i++) {
        if (fscanf(File, "%d %d %d %d", &From, &To, &Victim, &Alpha) != 4 ||
            From != i) {
            fprintf(stderr, "BetaTour hook malformed action row %d\n", i);
            exit(97);
        }
        BT_ValidateNoop(From, To, Victim);
        if (To == 0)
            continue;
        if (BT_ApplyOne(From, To, Victim, Alpha) != 0) {
            Rejected++;
            fprintf(stderr,
                    "BetaTour rejected swap from=%d to=%d victim=%d alpha=%d: %s\n",
                    From, To, Victim, Alpha, BT_LastSwapRejectReason);
        } else {
            BT_ActionTo[From] = To;
            BT_ActionVictim[From] = Victim;
            Swapped++;
        }
    }
    {
        char Extra;
        if (fscanf(File, " %c", &Extra) != EOF) {
            fprintf(stderr, "BetaTour action file has extra data\n");
            exit(97);
        }
    }
    fclose(File);
    if (Rejected) {
        fprintf(stderr, "BetaTour hook rejected %d candidate actions\n",
                Rejected);
        exit(97);
    }
    BT_WriteCandidates(Directory, "post.candidates");
    File = BT_OpenTemporary(Directory, "applied.stats", Final, Temporary);
    fprintf(File, "swapped %d\nrejected %d\nrow_length_preserved 1\nwait_wall_s %.6f\n",
            Swapped, Rejected, BT_WallTime() - Start);
    BT_Commit(File, Temporary, Final);
    BT_ObserveBetterTour(0);
    BT_WriteTelemetry(Directory, 1);
}

static void BT_ExportNativeCandidates(void)
{
    const char *Directory = BT_HookDirectory();
    if (Directory)
        BT_WriteCandidates(Directory, "native.candidates");
}

static void BT_AfterFirstTrial(void)
{
    const char *Directory = BT_HookDirectory();
    if (Directory && Trial == 1) {
        BT_WriteCandidates(Directory, "live.candidates");
        BT_WriteFirstTour(Directory); /* readiness marker; published last */
        BT_WaitAndApply(Directory);
    } else if (Directory && BT_ActionTo) {
        BT_TrialsCompletedAfterAction++;
        BT_WriteTelemetry(Directory,
                          BT_WallTime() - BT_LastFullTelemetryWall >= 5.0);
    }
}

#endif
