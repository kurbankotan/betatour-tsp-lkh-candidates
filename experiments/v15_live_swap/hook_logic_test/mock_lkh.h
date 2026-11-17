/* Minimal stand-in for the parts of LKH.h that the BetaTour v11 hook uses.
 * It lets the hook logic be compiled and unit-tested without LKH sources.
 * Field and function semantics follow LKH-3 (1-based node IDs, candidate
 * rows terminated by To == 0, AddCandidate appending to the row). */
#ifndef BETATOUR_MOCK_LKH_H
#define BETATOUR_MOCK_LKH_H

#include <stdlib.h>

typedef long long GainType;
typedef struct Node Node;
typedef struct Candidate {
    Node *To;
    int Cost;
    int Alpha;
} Candidate;
struct Node {
    int Id;
    Candidate *CandidateSet;
    Node *Pred, *Suc, *OldPred, *OldSuc;
};

enum Types { TSP, ATSP };
static int ProblemType = TSP;
static int Run = 1;
static int Dimension = 0, DimensionSaved = 0;
static int Trial = 0;
static Node *NodeSet = 0;
static int *BetterTour = 0;
static GainType BetterCost = 0;

static int MockCost(Node *A, Node *B)
{
    int D = A->Id - B->Id;
    return 100 * (D < 0 ? -D : D);
}
#define C(A, B) MockCost((A), (B))

static int AddCandidate(Node *From, Node *To, int Cost, int Alpha)
{
    int Count = 0;
    Candidate *NFrom;
    if (From == To)
        return 0;
    if (!From->CandidateSet)
        From->CandidateSet = (Candidate *) calloc(3, sizeof(Candidate));
    for (NFrom = From->CandidateSet; NFrom->To && NFrom->To != To; NFrom++)
        Count++;
    if (NFrom->To)
        return 0;
    NFrom->Cost = Cost;
    NFrom->Alpha = Alpha;
    NFrom->To = To;
    From->CandidateSet = (Candidate *)
        realloc(From->CandidateSet, (Count + 2) * sizeof(Candidate));
    From->CandidateSet[Count + 1].To = 0;
    return 1;
}

#endif
