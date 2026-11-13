/* BetaTour v10 delayed-barrier research hook for LKH 3.0.14. */
#ifndef BETATOUR_HOOK_H
#define BETATOUR_HOOK_H

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

static double BT_SearchStartWall = 0.0;
static int BT_BarrierTriggered = 0;

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

static void BT_WriteBarrierTour(const char *Directory)
{
    char Final[PATH_MAX], Temporary[PATH_MAX];
    FILE *File = BT_OpenTemporary(Directory, "barrier.tour", Final, Temporary);
    int i;
    fprintf(File, "%d\n", DimensionSaved);
    for (i = 1; i <= DimensionSaved; i++)
        fprintf(File, "%d\n", BetterTour[i]);
    BT_Commit(File, Temporary, Final);
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
}

static int BT_ApplyOne(int FromId, int ToId, int Slot, int Alpha,
                       int *WasExisting, int *WasMoved)
{
    Node *From, *To;
    Candidate *Start, *Chosen, Temp;
    int Count = 0, Index, Destination;

    if (FromId < 1 || FromId > DimensionSaved || ToId < 1 ||
        ToId > DimensionSaved || ToId == FromId || Slot < 0 ||
        Alpha < 0 || Alpha == INT_MAX)
        return -1;
    From = &NodeSet[FromId];
    To = &NodeSet[ToId];
    Start = From->CandidateSet;
    for (Chosen = Start; Chosen && Chosen->To; Chosen++)
        if (Chosen->To == To)
            *WasExisting = 1;
    /* A rank-only action must retain the native edge's alpha value. */
    if (!*WasExisting && !AddCandidate(From, To, C(From, To), Alpha))
        return -1;
    Start = From->CandidateSet;
    for (Chosen = Start; Chosen && Chosen->To; Chosen++)
        Count++;
    for (Index = 0; Index < Count; Index++)
        if (Start[Index].To == To)
            break;
    if (Index == Count)
        return -1;
    if (Slot > 0) {
        Destination = Slot - 1;
        if (Destination >= Count)
            Destination = Count - 1;
        if (Index != Destination) {
            int j;
            Temp = Start[Index];
            if (Index > Destination)
                for (j = Index; j > Destination; j--)
                    Start[j] = Start[j - 1];
            else
                for (j = Index; j < Destination; j++)
                    Start[j] = Start[j + 1];
            Start[Destination] = Temp;
            *WasMoved = 1;
        }
    }
    return 0;
}

static void BT_WaitAndApply(const char *Directory)
{
    char Action[PATH_MAX], Final[PATH_MAX], Temporary[PATH_MAX];
    FILE *File;
    int N, i, From, To, Slot, Alpha;
    int Added = 0, Existing = 0, Rejected = 0, Moved = 0;
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
    for (i = 1; i <= N; i++) {
        int WasExisting = 0, WasMoved = 0;
        if (fscanf(File, "%d %d %d %d", &From, &To, &Slot, &Alpha) != 4 ||
            From != i) {
            fprintf(stderr, "BetaTour hook malformed action row %d\n", i);
            exit(97);
        }
        if (To == 0)
            continue;
        if (BT_ApplyOne(From, To, Slot, Alpha, &WasExisting,
                        &WasMoved) != 0)
            Rejected++;
        else {
            Added += !WasExisting;
            Existing += WasExisting;
            Moved += WasMoved;
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
    fprintf(File, "added %d\nexisting %d\nmoved %d\nrejected %d\nwait_wall_s %.6f\n",
            Added, Existing, Moved, Rejected, BT_WallTime() - Start);
    BT_Commit(File, Temporary, Final);
}

static void BT_ExportNativeCandidates(void)
{
    const char *Directory = BT_HookDirectory();
    if (Directory) {
        BT_SearchStartWall = BT_WallTime();
        BT_WriteCandidates(Directory, "native.candidates");
    }
}

static void BT_AfterFirstTrial(void)
{
    const char *Directory = BT_HookDirectory();
    const char *AfterText = getenv("BETATOUR_HOOK_AFTER_S");
    double After = AfterText && atof(AfterText) > 0 ? atof(AfterText) : 0.0;
    if (Directory && !BT_BarrierTriggered && Trial >= 1 &&
        BT_WallTime() - BT_SearchStartWall >= After) {
        BT_BarrierTriggered = 1;
        BT_WriteCandidates(Directory, "live.candidates");
        BT_WriteBarrierTour(Directory); /* readiness marker; published last */
        BT_WaitAndApply(Directory);
    }
}

#endif
