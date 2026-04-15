Stable batch Bass fitting for Instagram Reels with p, q constrained to (0, 1).

Inputs
------
Dynamic:
  /Users/jimwu/Desktop/Bass_Model/Reels Data/reels_dynamic_info.csv
  Expected columns:
    reels_shortcode,views,plays,likes,comments,timestamp

Static:
  /Users/jimwu/Desktop/Bass_Model/Reels Data/reels_static_info.csv
  Expected columns:
    kol_account,reels_shortcode,post_time,duration,caption
  Only reels posted strictly after 2026-04-10 are used.

Selection rules
---------------
- Skip rows with missing views or timestamp.
- Use only reels with static post_time > 2026-04-10 23:59:59.
- Rank reels by usable data points after cleaning.
- Train at most 20 reels.

Major modeling changes
----------------------
- p and q are forced into (0, 1) with a logistic transform.
- M is forced to be > max(y) with a softplus transform.
- Fit uses interval growth (diffs) in log-space + a small cumulative-level term.
- Mild regularization keeps p and q away from unstable boundary solutions.

Outputs
-------
For each reel:
  /Users/jimwu/Desktop/Bass_Model/Output/<reels_shortcode>/
    - observed_vs_fit.csv
    - 30d_projection.csv
    - effective_M.csv
    - summary.json
    - fixed_M_bass_fit_30d.png
    - effective_M.png
    - effective_M_log.png

Global outputs:
  /Users/jimwu/Desktop/Bass_Model/Output/
    - selected_reels.csv
    - training_summary.csv
    - training_errors.csv
"""
