"""
scoring.py
----------
Converts concrete, verifiable inputs into the 0-100 coding_score and
communication_score features the model was trained on.

Why not just ask for a raw 0-100 number? Because "rate your coding skill
0-100" is meaningless to most students. Asking for actual problem counts
(LeetCode + Codeforces combined) and a few concrete communication questions
gives a number that's grounded in something real, while still landing in
the same 0-100 range the model expects -- so no retraining is needed.
"""


def compute_coding_score(easy: int, medium: int, hard: int) -> float:
    """
    Weighted, diminishing-returns score from DSA problems solved
    (LeetCode + Codeforces combined).

    Weights reflect typical difficulty: hard problems count for more than
    easy ones. A square-root curve is used instead of a raw linear sum so
    that early progress (0 -> 50 problems) moves the score more than the
    same jump does at the high end (500 -> 550) -- diminishing returns,
    similar to how interviewers view problem counts in practice.
    """
    weighted = easy * 1 + medium * 2.5 + hard * 5
    score = (weighted ** 0.5) * 2.75
    return round(min(100, score), 1)


def compute_communication_score(group_comfort: int, past_feedback: str, fluency: int) -> float:
    """
    Blends three self-assessed inputs into a single 0-100 score:
      - group_comfort: 1-5 scale (comfort in GDs / mock interviews)
      - past_feedback: category from actual past interview/presentation feedback
      - fluency: 1-5 scale (clarity explaining technical concepts out loud)

    Weights: comfort 40%, past feedback 30%, fluency 30% -- past feedback is
    weighted less than the two self-ratings combined since many students
    genuinely haven't had a formal interview yet ("No feedback yet").
    """
    feedback_map = {
        "No feedback yet": 50.0,   # neutral -- unknown, not penalized
        "Mostly negative": 25.0,
        "Mixed": 55.0,
        "Mostly positive": 85.0,
    }

    comfort_norm = (group_comfort - 1) / 4 * 100
    fluency_norm = (fluency - 1) / 4 * 100
    feedback_val = feedback_map.get(past_feedback, 50.0)

    score = 0.4 * comfort_norm + 0.3 * feedback_val + 0.3 * fluency_norm
    return round(min(100, max(0, score)), 1)


if __name__ == "__main__":
    # quick sanity checks
    print("Beginner coder (20 easy, 5 medium, 0 hard):", compute_coding_score(20, 5, 0))
    print("Intermediate (80 easy, 40 medium, 5 hard):", compute_coding_score(80, 40, 5))
    print("Advanced (250 easy, 150 medium, 40 hard):", compute_coding_score(250, 150, 40))
    print()
    print("Low comm (comfort=2, feedback=Mostly negative, fluency=2):",
          compute_communication_score(2, "Mostly negative", 2))
    print("Mid comm (comfort=3, feedback=No feedback yet, fluency=3):",
          compute_communication_score(3, "No feedback yet", 3))
    print("High comm (comfort=5, feedback=Mostly positive, fluency=5):",
          compute_communication_score(5, "Mostly positive", 5))
